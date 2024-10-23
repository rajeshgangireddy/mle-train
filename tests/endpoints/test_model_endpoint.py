import json
import unittest

import requests


class TestModelEndpoint(unittest.TestCase):
    def setUp(self):
        self.url = "http://127.0.0.1:5000/predict"
        self.headers = {"Content-Type": "application/json"}
        self.sample_data = [
            {
                "Temperature": 23.5,
                "Humidity": 30,
                "Light": 500,
                "CO2": 400,
                "HumidityRatio": 0.004,
            },
            {
                "Temperature": 22.0,
                "Humidity": 35,
                "Light": 450,
                "CO2": 420,
                "HumidityRatio": 0.005,
            },
        ]
        self.sample_output = [0, 0]

    def test_predict(self):
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(self.sample_data)
        )
        self.assertEqual(response.status_code, 200)
        predictions = response.json()
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.sample_data))
        self.assertEqual(predictions, self.sample_output)


if __name__ == "__main__":
    unittest.main()
