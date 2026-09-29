import unittest
from pathlib import Path

from tools.validate_dataset import (
    Results,
    date_token,
    load_rules,
    parse_weather_line,
    simulation_can_form_sequence,
)


class DatasetValidatorTest(unittest.TestCase):
    def test_simulation_sequence_requires_enough_frames_for_random_interval(self):
        self.assertFalse(simulation_can_form_sequence(6, input_length=3))
        self.assertTrue(simulation_can_form_sequence(7, input_length=3))

    def test_real_date_token_matches_dataset_loader(self):
        self.assertEqual(date_token("20230725.png"), "20230725")
        self.assertEqual(date_token("fire_20230725.jpg"), "20230725")
        self.assertIsNone(date_token("fire_today.png"))

    def test_weather_line_requires_paper_six_features(self):
        results = Results(detail_limit=20)
        values = parse_weather_line(
            "2023 7 25 0000 87.08 37 0.00 404.64 240 50",
            Path("weather.txt"),
            1,
            results,
            strict_single_space=True,
        )

        self.assertEqual(len(values), 10)
        self.assertEqual(results.error_count, 0)

    def test_sim_weather_rejects_spacing_that_breaks_current_loader(self):
        results = Results(detail_limit=20)
        values = parse_weather_line(
            "2023  7 25 0000 87.08 37 0.00 404.64 240 50",
            Path("weather.txt"),
            1,
            results,
            strict_single_space=True,
        )

        self.assertIsNone(values)
        self.assertEqual(results.issue_counts["ERROR:WEATHER_SPACING"], 1)

    def test_report_detail_limit_does_not_change_total_counts(self):
        results = Results(detail_limit=1)
        path = Path("unused")
        results.error("ONE", path, "one")
        results.error("TWO", path, "two")

        self.assertEqual(results.error_count, 2)
        self.assertEqual(len(results.issues), 1)

    def test_current_config_matches_environment_channel_count(self):
        results = Results(detail_limit=20)
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

        rules = load_rules(config_path, results)

        self.assertEqual(rules.modal_channel, 29)
        self.assertEqual(results.error_count, 0)


if __name__ == "__main__":
    unittest.main()
