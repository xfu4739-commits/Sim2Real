"""Extract Drive archives into NEW raw scene directories, with safe resumability.

Default is inventory only. --execute writes outputs. Original archive structure
and labels are preserved; this tool does not create scientific train/test splits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import zipfile

GIB = 1024 ** 3
PATTERN = re.compile(r"\d{4}_\d{5}\.(zip|rar|7z)$", re.I)
RESERVATION_LOCK = threading.Lock()
RESERVED_BYTES = 0
PRINT_LOCK = threading.Lock()


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def bounded_path(root, name):
    # ZIP member names may contain either Windows or POSIX separators.
    name = name.replace('\\', '/')
    if name.startswith('/') or ':' in name or '..' in name.split('/'):
        raise ValueError(f'Unsafe archive member: {name}')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path outside destination: {name}')
    return target


def find_7zip(configured):
    candidates = [configured, shutil.which('7z'), shutil.which('7zz'),
                  r'C:\Program Files\7-Zip\7z.exe',
                  r'C:\Program Files (x86)\7-Zip\7z.exe']
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def zip_extract(archive, work):
    for member in archive.infolist():
        target = bounded_path(work, member.filename)
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only atomically finished files from this tool are reused on resume.
        if target.is_file() and target.stat().st_size == member.file_size:
            continue
        part = target.with_name(target.name + '.extract-part')
        with archive.open(member) as source, part.open('wb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        # ZipFile verifies member CRC at EOF. Never mark a partial read complete.
        os.replace(part, target)


def list_members_7zip(binary, archive):
    result = subprocess.run([binary, 'l', '-slt', str(archive)],
                            capture_output=True, text=True, errors='replace')
    if result.returncode:
        raise RuntimeError(f'Cannot list archive: {result.stderr[-1000:]}')
    # Parse only member details after the separator, not archive header fields.
    parts = result.stdout.split('----------', 1)
    if len(parts) != 2:
        raise RuntimeError('Unrecognized 7-Zip listing')
    members = []
    for block in parts[1].split('\n\n'):
        if not block.strip():
            continue
        path_match = re.search(r'^Path = (.*)$', block, flags=re.M)
        if not path_match:
            continue
        name = path_match.group(1)
        bounded_path(Path.cwd(), name)
        is_dir = re.search(r'^Folder = \+$', block, flags=re.M) is not None
        size_match = re.search(r'^Size = (\d+)$', block, flags=re.M)
        size = int(size_match.group(1)) if size_match else 0
        members.append({'name': name, 'size': size, 'is_dir': is_dir})
    if not members:
        raise RuntimeError(f'No archive members found: {archive.name}')
    return members


def bytes_still_needed(work, members):
    total = sum(member['size'] for member in members if not member['is_dir'])
    existing = 0
    for member in members:
        if member['is_dir']:
            continue
        target = bounded_path(work, member['name'])
        if not target.is_file():
            continue
        actual = target.stat().st_size
        if actual == member['size']:
            existing += member['size']
            continue
        # 7-Zip -aos would skip a wrong-size partial file; remove it before resume.
        target.unlink()
    return total - existing


def seven_zip_extract_cmd(binary, archive, work, args):
    overwrite = '-aoa' if args.overwrite else '-aos'
    return [
        binary, 'x', str(archive), f'-o{work}',
        '-y', overwrite, f'-mmt={args.seven_zip_threads}',
        '-bb0', '-bsp0', '-bso0',
    ]


def extract_one(source, output_root, args, seven_zip):
    global RESERVED_BYTES
    scene = source.stem
    destination = output_root / scene
    marker = destination / '.extraction-complete.json'
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved.get('archive_bytes') == source.stat().st_size:
            return {'archive': source.name, 'status': 'skipped_complete'}
        return {'archive': source.name, 'status': 'failed',
                'error': 'Archive size changed since completion; existing output preserved'}
    if destination.exists():
        return {'archive': source.name, 'status': 'failed',
                'error': 'Unmarked destination already exists; refusing to overwrite'}
    work = output_root / '.extract-work' / scene
    work.mkdir(parents=True, exist_ok=True)
    reserved = 0
    archive = None
    try:
        if seven_zip:
            members = list_members_7zip(seven_zip, source)
            for member in members:
                bounded_path(work, member['name'])
            total_bytes = sum(member['size'] for member in members if not member['is_dir'])
            needed = bytes_still_needed(work, members)
        else:
            # Fallback when 7-Zip is unavailable: only plain ZIP is supported.
            with source.open('rb') as stream:
                magic = stream.read(8)
            if not magic.startswith(b'PK'):
                raise RuntimeError('Non-ZIP content (possibly RAR named .zip): install 7-Zip, then rerun')
            archive = zipfile.ZipFile(source)
            members = archive.infolist()
            for member in members:
                bounded_path(work, member.filename)
            total_bytes = sum(m.file_size for m in members if not m.is_dir())
            existing = sum(m.file_size for m in members if not m.is_dir()
                           and bounded_path(work, m.filename).is_file()
                           and bounded_path(work, m.filename).stat().st_size == m.file_size)
            needed = total_bytes - existing
        # Reserve capacity across workers. Add filesystem overhead margin.
        required = int(needed * 1.1) + 64 * 1024 ** 2
        with RESERVATION_LOCK:
            free = shutil.disk_usage(output_root).free
            if free - RESERVED_BYTES - required < args.reserve_gib * GIB:
                return {'archive': source.name, 'status': 'insufficient_space',
                        'estimated_needed_gib': round(required / GIB, 2)}
            RESERVED_BYTES += required
            reserved = required
        resume_note = 'resume' if 0 < needed < total_bytes else 'full'
        say(
            f'Extracting {source.name} ({resume_note}), '
            f'estimated additional {required/GIB:.2f} GiB, '
            f'7-Zip threads={args.seven_zip_threads}, workers={args.workers}'
        )
        if seven_zip:
            log = output_root / '.extract-work' / f'{scene}.7z.log'
            cmd = seven_zip_extract_cmd(seven_zip, source, work, args)
            with log.open('wb') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'7-Zip exit {result.returncode}; see {log}')
        else:
            zip_extract(archive, work)
        # Normalize an optional single outer directory, without copying data.
        children = list(work.iterdir())
        payload = work
        if len(children) == 1 and children[0].is_dir():
            nested = children[0]
            if (nested / 'Topography_Map').is_dir():
                payload = nested
        info = {'archive': source.name, 'archive_bytes': source.stat().st_size,
                'layout': 'raw author files; not training-loader layout'}
        (payload / '.extraction-complete.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        payload.rename(destination)
        if payload != work:
            work.rmdir()  # Nonrecursive; succeeds only for our now-empty staging folder.
        return {'archive': source.name, 'status': 'complete'}
    except Exception as error:
        return {'archive': source.name, 'status': 'failed', 'error': str(error)}
    finally:
        if archive:
            archive.close()
        if reserved:
            with RESERVATION_LOCK:
                RESERVED_BYTES -= reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-source', type=Path, default=Path('G:/我的云端硬盘'))
    parser.add_argument('--real-source', type=Path, default=Path(
        'G:/.shortcut-targets-by-id/1QJbNu35VzCCDOhkc3bn5guIoLEyA0cRF/Sim2Real-Fire_Dataset_real_data'))
    parser.add_argument('--output', type=Path, default=Path('C:/Users/10115/Datasets/Sim2Real-Fire'))
    parser.add_argument('--kind', choices=['sim', 'real', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel archives to extract at once (1-4).')
    parser.add_argument('--seven-zip-threads', dest='seven_zip_threads', default='on',
                        help='7-Zip -mmt value: on/off or an integer thread count.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Force full re-extract (-aoa). Default skips finished files (-aos).')
    parser.add_argument('--reserve-gib', type=float, default=20)
    parser.add_argument('--seven-zip')
    parser.add_argument('--limit', type=int, help='Limit packages per data kind for a trial')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    jobs = []
    # Finish the smaller real dataset first when both are selected.
    if args.kind in ['real', 'both']:
        jobs.append(('real', args.real_source, args.output / 'real_full'))
    if args.kind in ['sim', 'both']:
        jobs.append(('sim', args.sim_source, args.output / 'sim_full'))
    seven_zip = find_7zip(args.seven_zip)
    for kind, source_root, output_root in jobs:
        sources = sorted(p for p in source_root.iterdir() if p.is_file() and PATTERN.fullmatch(p.name))
        if args.limit:
            sources = sources[:args.limit]
        say(f'{kind}: {len(sources)} archives -> {output_root}; 7-Zip={seven_zip or "not found"}')
        if not args.execute:
            say('Inventory only. Add --execute to extract. Raw archive structure is preserved.')
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        # Submit only one small batch at a time: stop scheduling on low space.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(sources), args.workers):
                batch = [pool.submit(extract_one, src, output_root, args, seven_zip)
                         for src in sources[start:start+args.workers]]
                stop = False
                for future in as_completed(batch):
                    result = future.result()
                    say(json.dumps(result, ensure_ascii=False))
                    with (output_root / 'extraction-log.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stop |= result['status'] == 'insufficient_space'
                if stop:
                    say('Stopped: reserve disk space reached. Free space or use another --output, then rerun.')
                    break


if __name__ == '__main__':
    main()
