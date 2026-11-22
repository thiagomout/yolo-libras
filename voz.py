"""Síntese offline; pode ser testada sem câmera: python voz.py --text 'Letra bê'."""
from __future__ import annotations

import argparse
import base64
import platform
import queue
import shutil
import subprocess
import sys
import threading

LETTER_NAMES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (
    'á', 'bê', 'cê', 'dê', 'é', 'éfe', 'gê', 'agá', 'i', 'jota', 'cá', 'éle',
    'ême', 'êne', 'ó', 'pê', 'quê', 'érre', 'ésse', 'tê', 'u', 'vê',
    'dáblio', 'xis', 'ípsilon', 'zê')))
LETTER_NAMES['Ç'] = 'cê cedilha'


def letter_text(label: str) -> str:
    return f"Letra {LETTER_NAMES.get(label.upper(), label)}"


class Speaker:
    def __init__(self, backend: str = 'auto') -> None:
        self.command = shutil.which('espeak-ng') or shutil.which('espeak')
        self.powershell = shutil.which('powershell.exe')
        self.backend = ('espeak' if sys.platform.startswith('linux') and self.command else 'pyttsx3') if backend == 'auto' else backend
        if backend == 'auto' and 'microsoft' in platform.release().lower() and self.powershell:
            self.backend = 'windows'
        self.engine = None
        if self.backend == 'windows':
            if not self.powershell:
                raise RuntimeError('PowerShell do Windows não encontrado no PATH do WSL.')
        elif self.backend == 'espeak':
            if not self.command:
                raise RuntimeError('Instale espeak-ng: sudo apt install espeak-ng')
        else:
            import pyttsx3
            self.engine = pyttsx3.init()
            for voice in self.engine.getProperty('voices'):
                details = f"{voice.id} {voice.name} {getattr(voice, 'languages', [])}".lower()
                if any(tag in details for tag in ('portugu', 'brazil', 'pt-br', 'pt_br')):
                    self.engine.setProperty('voice', voice.id)
                    break
            self.engine.setProperty('rate', 155)
            self.engine.setProperty('volume', 1.0)

    def say(self, text: str) -> None:
        if self.backend == 'windows':
            escaped = text.replace("'", "''")
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$voice = $speaker.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'pt-BR' } | Select-Object -First 1; "
                      "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
                      f"$speaker.Speak('{escaped}'); $speaker.Dispose()")
            encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
            result = subprocess.run([self.powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], timeout=20, capture_output=True)
            if result.returncode:
                details = result.stderr.decode(errors='replace').strip()
                raise RuntimeError(f"Falha na voz do Windows: {details or 'PowerShell retornou erro'}")
        elif self.backend == 'espeak':
            result = subprocess.run([self.command, '-v', 'pt-br', '-s', '155', text], check=True, timeout=15, capture_output=True)
            error = result.stderr.decode(errors='replace')
            if any(marker in error.lower() for marker in ('cannot find card', 'unknown pcm', 'error:', 'failed')):
                raise RuntimeError(f"Saída de áudio indisponível: {error[-400:].strip()}")
        else:
            self.engine.say(text)
            self.engine.runAndWait()

    def close(self) -> None:
        if self.engine:
            self.engine.stop()


class SpeechWorker:
    """Mantém apenas a próxima fala, evitando acumular previsões antigas."""
    def __init__(self, backend: str = 'auto') -> None:
        self.backend = backend
        self.messages: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ready.wait(timeout=5)

    def say(self, label: str) -> bool:
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            return False
        self._replace(letter_text(label))
        return True

    def _replace(self, message: str | None) -> None:
        try:
            self.messages.get_nowait()
        except queue.Empty:
            pass
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            pass

    def _run(self) -> None:
        speaker = None
        try:
            speaker = Speaker(self.backend)
            print(f'Voz iniciada: {speaker.backend}')
            self.ready.set()
            while True:
                text = self.messages.get()
                if text is None:
                    break
                speaker.say(text)
        except Exception as exc:
            self.error = str(exc)
            print(f'Falha na voz: {exc}. Teste: python voz.py')
        finally:
            self.ready.set()
            if speaker:
                speaker.close()

    def close(self) -> None:
        self._replace(None)
        self.thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='Teste de voz. Letra á. Letra bê. Letra cê.')
    parser.add_argument('--backend', choices=['auto', 'espeak', 'pyttsx3', 'windows'], default='auto')
    args = parser.parse_args()
    speaker = None
    try:
        speaker = Speaker(args.backend)
        print(f'Sintetizador: {speaker.backend}')
        speaker.say(args.text)
        return 0
    except Exception as exc:
        print(f'Falha na síntese: {exc}')
        return 1
    finally:
        if speaker:
            speaker.close()


if __name__ == '__main__':
    raise SystemExit(main())
