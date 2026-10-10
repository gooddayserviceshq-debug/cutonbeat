"""Renders a real video from synthetic media and checks the files, not the exit code."""
import json, os, shutil, subprocess, sys, tempfile, unittest

from cutonbeat.synth import drum_track, write_wav, test_clip as make_clip

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-show_entries", "format=duration", "-of", "json", p], capture_output=True, text=True).stdout
    return json.loads(out)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestEndToEnd(unittest.TestCase):
    def test_render(self):
        d = tempfile.mkdtemp(prefix="cutonbeat-")
        song = write_wav(os.path.join(d, "song.wav"), drum_track(120, 40))
        clips = os.path.join(d, "clips")
        os.makedirs(clips)
        for i, pat in enumerate(["testsrc2", "smptebars", "rgbtestsrc", "testsrc"]):
            make_clip(os.path.join(clips, f"c{i}.mp4"), pattern=pat, hue=i * 60)
        with open(os.path.join(clips, "clips.json"), "w") as fh:
            json.dump({"c0.mp4": "quiet", "c1.mp4": "chorus"}, fh)
        out = os.path.join(d, "out")
        r = subprocess.run([sys.executable, "-m", "cutonbeat", "--song", song, "--clips", clips, "--out", out,
                            "--title", "Test Pattern", "--artist", "Synth", "--end-text", "@example",
                            "--short-seconds", "8"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

        report = json.load(open(os.path.join(out, "report.json")))
        self.assertEqual(report["problems"], [])
        self.assertAlmostEqual(report["bpm"], 120, delta=1)

        m = probe(os.path.join(out, "master_16x9.mp4"))
        self.assertAlmostEqual(float(m["format"]["duration"]), 40, delta=0.25)
        v = [s for s in m["streams"] if s["codec_type"] == "video"][0]
        self.assertEqual((v["width"], v["height"]), (1920, 1080))
        self.assertIn("audio", [s["codec_type"] for s in m["streams"]])

        for n in (1, 2, 3):
            s = probe(os.path.join(out, f"short_{n}_9x16.mp4"))
            v = [x for x in s["streams"] if x["codec_type"] == "video"][0]
            self.assertEqual((v["width"], v["height"]), (1080, 1920))
            self.assertAlmostEqual(float(s["format"]["duration"]), 8, delta=0.1, msg=f"short_{n} length")
        self.assertTrue(os.path.getsize(os.path.join(out, "contact_sheet.jpg")) > 10_000)
        self.assertGreaterEqual(len(report["thumbs"]), 1)

        cuts = json.load(open(os.path.join(out, "cuts.json")))
        self.assertEqual({c["clip"] for c in cuts if c["tier"] == 0} & {"c1.mp4"}, set(), "chorus clip in a quiet cut")
        self.assertEqual({c["clip"] for c in cuts if c["tier"] == 2} & {"c0.mp4"}, set(), "quiet clip in a loud cut")
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
