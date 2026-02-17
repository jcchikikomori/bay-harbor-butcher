import os
import tempfile
import unittest
from unittest import mock

import app


class TestApp(unittest.TestCase):
    def test_get_next_output_path_increments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = os.path.join(temp_dir, "output_1.png")
            second = os.path.join(temp_dir, "output_2.png")
            open(first, "a", encoding="utf-8").close()
            open(second, "a", encoding="utf-8").close()

            next_path = app.get_next_output_path(temp_dir)

            self.assertTrue(next_path.endswith("output_3.png"))

    def test_parse_args_uses_env_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"SD_PROMPT": "hello from env"}, clear=False):
            with mock.patch("sys.argv", ["app.py"]):
                args = app.parse_args()

        self.assertEqual(args.prompt, "hello from env")

    def test_parse_args_uses_env_defaults(self) -> None:
        env = {
            "SD_IMAGE_HEIGHT": "640",
            "SD_IMAGE_WIDTH": "768",
            "SD_MODEL": "my-model",
            "SD_STRENGTH": "0.7",
            "SD_GUIDANCE_SCALE": "8.5",
            "SD_NUM_INFERENCE_STEPS": "25",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("sys.argv", ["app.py"]):
                args = app.parse_args()

        self.assertEqual(args.height, 640)
        self.assertEqual(args.width, 768)
        self.assertEqual(args.model, "my-model")
        self.assertEqual(args.strength, 0.7)
        self.assertEqual(args.guidance_scale, 8.5)
        self.assertEqual(args.num_inference_steps, 25)


if __name__ == "__main__":
    unittest.main()
