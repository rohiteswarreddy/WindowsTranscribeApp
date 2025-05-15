import os
import sys
import tempfile
import logging
import numpy as np
import whisper
import noisereduce as nr
import soundfile as sf
from logging.handlers import RotatingFileHandler
from pydub import AudioSegment
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QTextEdit, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QProgressBar, QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

import magic
import nltk  

try:
    nltk.data.find("tokenizers/punkt")
    logging.info("NLTK 'punkt' tokenizer found.")
except LookupError:
    logging.warning("NLTK 'punkt' tokenizer not found. Please install it manually using nltk.download('punkt') before running this application.")
    raise RuntimeError("Missing NLTK data. Run: import nltk; nltk.download('punkt')")



log_file = "transcriber.log"
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler]
)


class QtSignalLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class AudioProcessor:
    def __init__(self):
        self.model = whisper.load_model("small")
        logging.info("Whisper model loaded successfully.")

    def is_valid_audio_file(self, file_path):
        allowed_mime_types = [
            'audio/wav', 'audio/x-wav', 'audio/mpeg',
            'audio/flac', 'audio/x-flac', 'audio/mp4'
        ]
        mime_type = magic.from_file(file_path, mime=True)
        is_valid = mime_type in allowed_mime_types
        logging.info(f"Checked MIME type: {mime_type}, Valid: {is_valid}")
        return is_valid

    def convert_to_wav(self, file_path):
    # Validate file type before processing
        if not self.is_valid_audio_file(file_path):
            logging.error(f"Invalid file type attempted: {file_path}")
            raise ValueError("Invalid or unsupported audio file type.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            logging.info(f"File already in WAV format: {file_path}")
            return file_path

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()  # Ensure the file is closed before writing

        try:
            logging.info(f"Converting file to WAV: {file_path} -> {tmp_path}")
            
            try:
                audio = AudioSegment.from_file(file_path)
                audio.export(tmp_path, format="wav")
            except Exception as e:
                logging.error(f"Audio conversion failed: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            return tmp_path
        except Exception as e:
            logging.error(f"Error converting file: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    def transcribe(self, file_path, apply_noise=True, language="en"):
        wav_path = self.convert_to_wav(file_path)

        if apply_noise:
            data, sr = sf.read(wav_path)
            if data.ndim == 2:
                data = np.mean(data, axis=1)
            data = nr.reduce_noise(y=data, sr=sr)
            sf.write(wav_path, data, sr)

        logging.info("Starting Whisper transcription...")
        result = self.model.transcribe(wav_path, language=language)
        return result["text"]


class TranscriptionThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor, file_path, apply_noise):
        super().__init__()
        self.processor = processor
        self.file_path = file_path
        self.apply_noise = apply_noise
        self._running = True  # ✅ Control flag for thread cancellation

    def run(self):
        try:
            if not self._running:
                logging.info("Transcription aborted before start.")
                return

            text = self.processor.transcribe(self.file_path, self.apply_noise)

            if not self._running:
                logging.info("Transcription aborted after processing.")
                return

            self.finished.emit(text)
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            self.error.emit(str(e))

    def stop(self):
        logging.info("Transcription stop requested.")
        self._running = False



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized Transcriber & Summarizer")
        self.setGeometry(100, 100, 1200, 700)
        self.processor = AudioProcessor()

        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(300)

        self.transcription_output = QTextEdit()
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)

        self.progress_bar = QProgressBar()
        self.noise_checkbox = QCheckBox("Apply Noise Reduction")
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.clicked.connect(self.start_transcription)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.summarize_btn = QPushButton("Summarize Transcript")
        self.summarize_btn.clicked.connect(self.summarize_text)

        self.save_transcript_btn = QPushButton("Save Transcript")
        self.save_transcript_btn.clicked.connect(self.save_transcript)

        self.save_summary_btn = QPushButton("Save Summary")
        self.save_summary_btn.clicked.connect(self.save_summary)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Transcript"))
        controls_layout.addWidget(self.transcription_output)
        controls_layout.addWidget(QLabel("Summary"))
        controls_layout.addWidget(self.summary_output)
        controls_layout.addWidget(QLabel("Log Output"))
        controls_layout.addWidget(self.log_output)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.noise_checkbox)
        controls_layout.addWidget(self.transcribe_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.summarize_btn)
        controls_layout.addWidget(self.save_transcript_btn)
        controls_layout.addWidget(self.save_summary_btn)

        right_panel = QWidget()
        right_panel.setLayout(controls_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_list)
        splitter.addWidget(right_panel)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.addWidget(splitter)

        add_file_btn = QPushButton("Add Audio File(s)")
        add_file_btn.clicked.connect(self.add_files)
        main_layout.addWidget(add_file_btn)

        self.setCentralWidget(container)

        self.log_handler = QtSignalLogHandler()
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.log_handler.log_signal.connect(self.append_log)
        logging.getLogger().addHandler(self.log_handler)

    def append_log(self, message):
        self.log_output.append(message)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        self.file_list.addItems(files)

    def start_transcription(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        file_path = selected_items[0].text()
        self.stop_btn.setEnabled(True)
        self.save_transcript_btn.setEnabled(False)
        self.save_summary_btn.setEnabled(False)
        self.summarize_btn.setEnabled(False)
        self.file_list.setEnabled(False)
        self.thread = TranscriptionThread(self.processor, file_path, self.noise_checkbox.isChecked())
        self.thread.finished.connect(self.display_transcription)
        self.thread.error.connect(self.display_error)
        self.thread.start()

    def display_transcription(self, text):
        self.transcription_output.setText(text)
        self.summarize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_transcript_btn.setEnabled(True)
        self.save_summary_btn.setEnabled(True)
        self.file_list.setEnabled(True)

    def display_error(self, error_msg):
        self.transcription_output.setText(f"Error: {error_msg}")
        self.stop_btn.setEnabled(False)
        self.file_list.setEnabled(True)
        

    def summarize_text(self):
        raw_text = self.transcription_output.toPlainText()
        if not raw_text.strip():
            self.summary_output.setPlainText("Transcript is empty.")
            return

        try:
            parser = PlaintextParser.from_string(raw_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary_sentences = summarizer(parser.document, 3)
            summary = "\n".join(str(sentence) for sentence in summary_sentences)
            self.summary_output.setPlainText(summary if summary else "No summary could be generated.")
        except Exception as e:
            self.summary_output.setPlainText(f"Summary failed: {e}")

    def save_transcript(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", "transcription.txt", "Text Files (*.txt)")
        if file_path:
            if os.path.exists(file_path):
                reply = QMessageBox.question(
                    self,
                    "Overwrite File?",
                    f"The file '{os.path.basename(file_path)}' already exists. Do you want to overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    logging.info("User canceled file overwrite.")
                    return

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.transcription_output.toPlainText())

            summary = self.summary_output.toPlainText().strip()
            if summary:
                summary_path = file_path.replace(".txt", "_summary.txt")
                if os.path.exists(summary_path):
                    reply = QMessageBox.question(
                        self,
                        "Overwrite Summary?",
                        f"The file '{os.path.basename(summary_path)}' already exists. Do you want to overwrite it?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        logging.info("User canceled summary overwrite.")
                        return
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)


    def save_summary(self):
        text = self.summary_output.toPlainText()
        self.save_text_to_file(text, "Save Summary")

    def save_text_to_file(self, text, dialog_title):
        if not text.strip():
            return

        file_path, _ = QFileDialog.getSaveFileName(self, dialog_title, "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
