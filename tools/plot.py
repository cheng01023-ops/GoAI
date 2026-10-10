#!/usr/bin/env python3
"""Draw the GoAI training curves as a PNG using only the Python standard library.

usage: python3 tools/plot.py runs/train_log.csv runs/training_curve.png
"""
import csv, struct, sys, zlib

FONT = {
 '0':["01110","10001","10011","10101","11001","10001","01110"],
 '1':["00100","01100","00100","00100","00100","00100","01110"],
 '2':["01110","10001","00001","00010","00100","01000","11111"],
 '3':["11111","00010","00100","00010","00001","10001","01110"],
 '4':["00010","00110","01010","10010","11111","00010","00010"],
 '5':["11111","10000","11110","00001","00001","10001","01110"],
 '6':["00110","01000","10000","11110","10001","10001","01110"],
 '7':["11111","00001","00010","00100","01000","01000","01000"],
 '8':["01110","10001","10001","01110","10001","10001","01110"],
 '9':["01110","10001","10001","01111","00001","00010","01100"],
 'A':["01110","10001","10001","11111","10001","10001","10001"],
 'B':["11110","10001","10001","11110","10001","10001","11110"],
 'C':["01110","10001","10000","10000","10000","10001","01110"],
 'D':["11100","10010","10001","10001","10001","10010","11100"],
 'E':["11111","10000","10000","11110","10000","10000","11111"],
 'F':["11111","10000","10000","11110","10000","10000","10000"],
 'G':["01110","10001","10000","10111","10001","10001","01111"],
 'H':["10001","10001","10001","11111","10001","10001","10001"],
 'I':["01110","00100","00100","00100","00100","00100","01110"],
 'J':["00111","00010","00010","00010","00010","10010","01100"],
 'K':["10001","10010","10100","11000","10100","10010","10001"],
 'L':["10000","10000","10000","10000","10000","10000","11111"],
 'M':["10001","11011","10101","10101","10001","10001","10001"],
 'N':["10001","10001","11001","10101","10011","10001","10001"],
 'O':["01110","10001","10001","10001","10001","10001","01110"],
 'P':["11110","10001","10001","11110","10000","10000","10000"],
 'Q':["01110","10001","10001","10001","10101","10010","01101"],
 'R':["11110","10001","10001","11110","10100","10010","10001"],
 'S':["01111","10000","10000","01110","00001","00001","11110"],
 'T':["11111","00100","00100","00100","00100","00100","00100"],
 'U':["10001","10001","10001","10001","10001","10001","01110"],
 'V':["10001","10001","10001","10001","10001","01010","00100"],
 'W':["10001","10001","10001","10101","10101","11011","10001"],
 'X':["10001","10001","01010","00100","01010","10001","10001"],
 'Y':["10001","10001","01010","00100","00100","00100","00100"],
 'Z':["11111","00001","00010","00100","01000","10000","11111"],
 ' ':["00000"]*7,
 '.':["00000","00000","00000","00000","00000","01100","01100"],
 '%':["11001","11010","00010","00100","01000","01011","10011"],
 '-':["00000","00000","00000","11111","00000","00000","00000"],
 '(':["00010","00100","01000","01000","01000","00100","00010"],
 ')':["01000","00100","00010","00010","00010","00100","01000"],
 ':':["00000","01100","01100","00000","01100","01100","00000"],
}

class Canvas:
    def __init__(self, w, h, bg=(255,255,255)):
        self.w, self.h = w, h
        self.px = bytearray(bg * (w*h))
    def dot(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y*self.w + x) * 3
            self.px[i:i+3] = bytes(c)
    def rect(self, x0, y0, x1, y1, c, fill=False):
        for y in range(max(0,y0), min(self.h-1,y1)+1):
            for x in range(max(0,x0), min(self.w-1,x1)+1):
                if fill or x in (x0,x1) or y in (y0,y1):
                    self.dot(x, y, c)
    def line(self, x0, y0, x1, y1, c, thick=1):
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            for oy in range(thick):
                for ox in range(thick):
                    self.dot(x0+ox, y0+oy, c)
            if x0 == x1 and y0 == y1: break
            e2 = 2*err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
    def text(self, x, y, s, c, scale=2):
        cx = x
        for ch in s.upper():
            g = FONT.get(ch, FONT[' '])
            for ry, row in enumerate(g):
                for rx, v in enumerate(row):
                    if v == '1':
                        for sy in range(scale):
                            for sx in range(scale):
                                self.dot(cx+rx*scale+sx, y+ry*scale+sy, c)
            cx += 6*scale
    def text_w(self, s, scale=2): return len(s)*6*scale
    def save(self, path):
        raw = b''.join(b'\x00' + bytes(self.px[y*self.w*3:(y+1)*self.w*3]) for y in range(self.h))
        def chunk(t, d):
            c = struct.pack('>I', len(d)) + t + d
            return c + struct.pack('>I', zlib.crc32(t+d) & 0xffffffff)
        png = (b'\x89PNG\r\n\x1a\n'
               + chunk(b'IHDR', struct.pack('>IIBBBBB', self.w, self.h, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(raw, 9))
               + chunk(b'IEND', b''))
        open(path, 'wb').write(png)

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def plot(csv_path, out_path):
    rows = load(csv_path)
    if not rows:
        print('no data'); return 1
    has_own = 'own_loss' in rows[0]          # 第 ⑤ 步的 CSV 才有这一列
    W, H = 980, 1360 if has_own else 1020
    cv = Canvas(W, H, (255,255,255))
    total_games = 0

    # ---- panels ----
    def panel(y0, y1, title, series, ymin, ymax, ylabel, fmt):
        L, R = 120, W-70
        T, B = y0+86, y1-70
        cv.rect(L, T, R, B, (60,60,60))
        for k in range(5):
            yy = T + (B-T)*k//4
            cv.line(L, yy, R, yy, (225,225,225))
            val = ymax - (ymax-ymin)*k/4.0
            lbl = fmt(val)
            cv.text(L-16-cv.text_w(lbl, 2), yy-7, lbl, (70,70,70), 2)
        n = len(rows)
        for i in range(n):
            xx = L + (R-L)*i//max(1, n-1)
            cv.line(xx, T, xx, B, (240,240,240))
            cv.text(xx-5, B+12, str(i+1), (70,70,70), 2)
        for name, key, color in series:
            pts = []
            for i, r in enumerate(rows):
                xx = L + (R-L)*i//max(1, n-1)
                v = r[key]
                yy = B - int((v-ymin)/(ymax-ymin)*(B-T))
                pts.append((xx, yy))
            for (x0,y0_),(x1,y1_) in zip(pts, pts[1:]):
                cv.line(x0, y0_, x1, y1_, color, 2)
            for (x, y) in pts:
                cv.rect(x-3, y-3, x+3, y+3, color, fill=True)
        cv.text(L, y0+14, title, (20,20,20), 2)
        cv.text(L, y0+42, ylabel, (90,90,90), 2)
        # legend, right aligned on its own line
        widths = [22 + cv.text_w(name, 2) + 30 for name, _, _ in series]
        lx = R - sum(widths)
        for (name, key, color), wdt in zip(series, widths):
            cv.rect(lx, y0+44, lx+16, y0+54, color, fill=True)
            cv.text(lx+22, y0+40, name, color, 2)
            lx += wdt
        cv.text(L, B+34, 'ITERATION', (70,70,70), 2)

    panel(0, 340, 'GOAI TRAINING - WIN RATE VS RANDOM',
          [('WIN RATE', 'winrate_vs_random', (200,40,40))],
          0, 1.0, 'FRACTION OF GAMES WON', lambda v: '%d%%' % round(v*100))
    panel(340, 680, 'GOAI TRAINING - POLICY LOSS',
          [('POLICY', 'policy_loss', (40,90,200))],
          0, max(0.5, max(r['policy_loss'] for r in rows)*1.1), 'CROSS ENTROPY',
          lambda v: '%.2f' % v)
    panel(680, 1020, 'GOAI TRAINING - VALUE LOSS',
          [('VALUE', 'value_loss', (20,150,80))],
          0, max(0.5, max(r['value_loss'] for r in rows)*1.3),
          'MSE (SCALAR HEAD) / CROSS ENTROPY (DIST HEAD)',
          lambda v: '%.2f' % v)
    if has_own:
        panel(1020, 1360, 'GOAI TRAINING - OWNERSHIP LOSS (STEP 5)',
              [('OWN', 'own_loss', (150,90,20))],
              0, max(0.5, max(r['own_loss'] for r in rows)*1.3), 'BCE, MEAN OVER POINTS',
              lambda v: '%.2f' % v)

    # fix: the y axis of panel 1 is a ratio, draw its labels as percentages
    cv.save(out_path)
    print('wrote', out_path)
    return 0

if __name__ == '__main__':
    sys.exit(plot(sys.argv[1] if len(sys.argv) > 1 else 'runs/train_log.csv',
                  sys.argv[2] if len(sys.argv) > 2 else 'runs/training_curve.png'))
