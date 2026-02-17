import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
