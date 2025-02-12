import os
import subprocess
import time
import lzma

# Run locally or under Kaggle submissions
LZMA_PATH = './main.out.xz'
if not os.path.exists(LZMA_PATH):
    LZMA_PATH = '/kaggle_simulations/agent/main.out.xz'

ENGINE_PATH = '/kaggle/working/engine.out'

with lzma.open(LZMA_PATH, 'rb') as lzma_file:
    with open(ENGINE_PATH, 'wb') as fout:
        fout.write(lzma_file.read())
    os.system('chmod +x %s' % (ENGINE_PATH))

class Engine:

    def __init__(self, file):

        self.eng = subprocess.Popen(
            [file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )

        self.best   = None
        self.ponder = None
        self.pos    = None

    def write(self, string):
        self.eng.stdin.write(string)
        self.eng.stdin.flush()

    def read_bestmove_line(self):

        while True:
            line       = self.eng.stdout.readline().strip()
            tokens     = line.split()

            has_best   = len(tokens) >= 2 and tokens[0] == 'bestmove'
            has_ponder = len(tokens) >= 4 and tokens[2] == 'ponder'

            best       = tokens[1] if has_best   else None
            ponder     = tokens[3] if has_ponder else None

            if best:
                return best, ponder

    def search(self, fen, prev_move, clock):

        start = time.time() * 1000

        ### House Keeping ###

        # Very first time for us being told to search
        if not self.pos:
            self.pos = 'position fen %s moves' % (fen);

        # Not the first time, and we have a move
        elif prev_move:
            self.pos += ' %s' % (prev_move)

        ### Conduct a search, which might be complex ###

        # Pondering, but did not have a ponderhit. Shutdown search.
        if self.ponder and self.ponder != prev_move:
            self.write('stop\n')
            self.read_bestmove_line() # No-op, wait until engine is done
            self.ponder = None

        # Pondering, and had a ponderhit. Finish search
        if self.ponder and self.ponder == prev_move:
            self.write('ponderhit\n')
            self.best, self.ponder = self.read_bestmove_line()
            self.ponder = None

        # Not pondering, start and finished a new search for the given FEN
        else:
            self.write('%s\ngo time %d\n' % (self.pos, clock))
            self.best, self.ponder = self.read_bestmove_line()

        # Update the position with what we intend to play
        self.pos += ' %s' % (self.best)

        ### By now we have a best move -- we now attempt to Ponder ###

        # Got "bestmove xxxx", without "ponder yyyy"
        if not self.ponder:
            return self.best

        # Start pondering, but don't wait to read any responses
        search_time = clock - (time.time() * 1000 - start)
        self.write('%s %s\ngo time %d ponder\n' % (self.pos, self.ponder, search_time))
        return self.best


# Create the Engine, which will persist between many calls to main()
engine = Engine(ENGINE_PATH)

def main(obs):
    fen       = obs.board
    time_left = obs.remainingOverageTime * 1000
    prev_move = obs.lastMove
    move      = engine.search(fen, prev_move, time_left)
    return move