"""The reading edition must work through the optional server, not only file://."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import server


class ReadingAssets(unittest.TestCase):
    def test_allowlisted_assets_match_files_and_have_correct_types(self):
        # No lifespan needed: static assets must not load/train the model.
        client = TestClient(server.app)
        for name in server.READING_ASSETS:
            with self.subTest(name=name):
                response = client.get('/' + name)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, (server.APP_HTML.parent / name).read_bytes())
                expected = 'text/css' if name.endswith('.css') else 'javascript'
                self.assertIn(expected, response.headers['content-type'])

    def test_repository_files_and_traversal_are_not_exposed(self):
        client = TestClient(server.app)
        for path in ['/server.py', '/course-engine.test.js', '/reading-unknown.js',
                     '/runs/latest.weights.pt', '/%2e%2e%2fserver.py', '/.git/config']:
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 404)

    def test_missing_asset_returns_a_controlled_404(self):
        with tempfile.TemporaryDirectory(prefix='glassbox-assets-') as directory:
            with patch.object(server, 'APP_HTML', Path(directory) / 'index.html'):
                self.assertEqual(TestClient(server.app).get('/reading.js').status_code, 404)


if __name__ == '__main__':
    unittest.main()
