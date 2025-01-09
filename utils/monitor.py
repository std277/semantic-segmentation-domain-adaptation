import time
import math
import sys

class Monitor:
    def __init__(self, file_name=None, resume=False, max_len=120):
        self.times = []
        self.monitor_id = 0
        self.max_len = max_len

        self.file = None
        if file_name is not None:
            self.file = open(file_name, 'a' if resume else 'w')
            if resume:
                self.file.write(f"\n{'_'*max_len}\nResumed\n\n")

    def start(self, desc, max_progress):
        self.desc = desc
        self.max_progress = max_progress

        self.time = time.time()
        self.last_update_time = self.time

        print(self.desc + ":")

    def stop(self):
        print()
        print()
        elapsed_time = time.time() - self.time
        self.times.append(elapsed_time)
        self.monitor_id =+ 1

    def update(self, progress, **kwargs):
        if progress == self.max_progress or (time.time() - self.last_update_time > 1.0):
            self._update_line(progress, **kwargs)


    # def _update_line(self, progress, **kwargs):
    #     # print(" " * self.max_len, end="\r")

    #     i = progress/self.max_progress

    #     bar_length = 10
    #     filled_length = int(math.floor(bar_length * i))
    #     bar = '█' * filled_length + '-' * (bar_length - filled_length)
    #     progress_line = f"Progress: |{bar}| {round(i*100): 3}% "

    #     elapsed_time = time.time() - self.time
    #     hours = int(elapsed_time // 3600)
    #     minutes = int((elapsed_time % 3600) // 60)
    #     seconds = int(elapsed_time % 60)
    #     time_line = f"\tTime: {hours:02}:{minutes:02}:{seconds:02} s "

    #     stats_line = ""
    #     for k, v in kwargs.items():
    #         stats_line += f"\t{k.replace('_', ' ').capitalize()}: {v} "

    #     print(progress_line + time_line + stats_line, end="\r", flush=True)

    #     if self.file and progress == self.max_progress:
    #         line = f"{self.desc}:\nTime: {hours:02}:{minutes:02}:{seconds:02} s \t"
    #         for k, v in kwargs.items():
    #             line += f"\t{k.replace('_', ' ').capitalize()}: {v} "
    #         line += "\n\n"
    #         self.file.write(line)
    #         self.file.flush()

    def _update_line(self, progress, **kwargs):
        # sys.stdout.write(" " * self.max_len + "\r")  # Clear the line

        i = progress / self.max_progress
        bar_length = 10
        filled_length = int(math.floor(bar_length * i))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        progress_line = f"Progress: |{bar}| {round(i*100): 3}% "

        elapsed_time = time.time() - self.time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_line = f"\tTime: {hours:02}:{minutes:02}:{seconds:02} s "

        stats_line = ""
        for k, v in kwargs.items():
            stats_line += f"\t{k.replace('_', ' ').capitalize()}: {v} "

        sys.stdout.write(progress_line + time_line + stats_line + "\r")
        sys.stdout.flush()

        if self.file and progress == self.max_progress:
            line = f"{self.desc}:\nTime: {hours:02}:{minutes:02}:{seconds:02} s \t"
            for k, v in kwargs.items():
                line += f"\t{k.replace('_', ' ').capitalize()}: {v} "
            line += "\n\n"
            self.file.write(line)
            self.file.flush()


    def print_stats(self):
        elapsed_time = sum(self.times)

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        line = f"Total elapsed time: {hours:02}:{minutes:02}:{seconds:02} s"
        
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def log(self, line):
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def __del__(self):
        if self.file:
            self.file.close()