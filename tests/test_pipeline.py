import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reconhecimento import StableRecognition
from treinar import find_class_parent, prepare_classification_dataset
from voz import Speaker, letter_text


class PipelineTests(unittest.TestCase):
    def test_hold_transition_and_repeat_after_pause(self):
        recognition = StableRecognition(hold=0.6, release=0.5)
        self.assertEqual(recognition.update('A', 0), (None, False))
        self.assertEqual(recognition.update('A', 0.7), ('A', True))
        self.assertEqual(recognition.update('A', 1), ('A', False))
        self.assertEqual(recognition.update('B', 1.1), (None, False))
        self.assertEqual(recognition.update('B', 1.8), ('B', True))
        recognition.update(None, 2)
        recognition.update('B', 2.7)
        self.assertEqual(recognition.update('B', 3.4), ('B', True))

    def test_short_pause_does_not_repeat(self):
        r = StableRecognition()
        r.update('A', 0)
        r.update('A', 1)
        r.update(None, 1.1)
        r.update('A', 1.2)
        self.assertEqual(r.update('A', 2), ('A', False))

    def test_split_dedup_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source' / 'train'
            for letter in 'ABCDEFGHIJ':
                folder = source / letter
                folder.mkdir(parents=True)
                for index in range(4):
                    (folder / f'{index}.png').write_bytes(f'{letter}:{index}'.encode())
                (folder / 'duplicate.png').write_bytes(f'{letter}:0'.encode())
                test = source.parent / 'test' / letter
                test.mkdir(parents=True)
                (test / 'independent.png').write_bytes(f'{letter}:test'.encode())
                (test / 'leaked.png').write_bytes(f'{letter}:1'.encode())
            self.assertEqual(find_class_parent(source), source)
            out = root / 'output'
            labels = prepare_classification_dataset(source, out, False)
            report = json.loads((out / 'split_manifest.json').read_text())
            self.assertEqual(report['duplicates_removed'], 20)
            self.assertTrue(all(count == 1 for count in report['counts']['test'].values()))
            self.assertEqual(prepare_classification_dataset(source, out, False), labels)
            self.assertTrue((source / 'A' / '0.png').exists())

    def test_audio_error_is_not_silent(self):
        from subprocess import CompletedProcess
        with patch('voz.shutil.which', return_value='/usr/bin/espeak-ng'):
            speaker = Speaker('espeak')
        with patch('voz.subprocess.run', return_value=CompletedProcess([], 0, b'', b'Unknown PCM default')):
            with self.assertRaisesRegex(RuntimeError, 'udio indispon'):
                speaker.say('Teste')
        self.assertEqual(letter_text('B'), 'Letra bê')


if __name__ == '__main__':
    unittest.main()
