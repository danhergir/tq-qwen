from __future__ import annotations

import json
import unittest

from tq_qwen_common import clean_response, encode_ndjson


class CommonHelpersTest(unittest.TestCase):
    def test_clean_response_strips_sentinels(self):
        raw = "TurboQuant KV cache is active.<|endoftext|><|im_start|>user"
        self.assertEqual(clean_response(raw), "TurboQuant KV cache is active.")

    def test_encode_ndjson_emits_single_line(self):
        payload = {"type": "token", "text": "hi"}
        encoded = encode_ndjson(payload)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded.decode("utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
