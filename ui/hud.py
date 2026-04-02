import tkinter as tk
import math, random, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.brain import JarvisBrain

BG     = "#020608"
CYAN   = "#00d4ff"
CDIM   = "#003344"
AMBER  = "#ff9800"
RED    = "#ff3333"
PURPLE = "#b040ff"
DARK   = "#040d14"
GREEN  = "#00ff88"

STATUS_COLOR = {
    "standby":     (CDIM,   False),
    "calibrating": (AMBER,  True),
    "listening":   (CYAN,   True),
    "processing":  (AMBER,  True),
    "speaking":    (PURPLE, True),
    "error":       (RED,    False),
}
STATUS_TEXT = {
    "standby":     ('SAY  "JARVIS"  TO ACTIVATE', CDIM),
    "calibrating": ("CALIBRATING...",              AMBER),
    "listening":   ("LISTENING...",                CYAN),
    "processing":  ("AI THINKING...",              AMBER),
    "speaking":    ("SPEAKING...",                 PURPLE),
    "error":       ("ERROR",                       RED),
}


class JarvisHUD:
    W = H = 700

    def __init__(self, brain: "JarvisBrain", face_auth=None):
        self.brain     = brain
        self._face_auth = face_auth
        self._status = "standby"
        self._color  = CDIM
        self._active = False
        self._angle  = 0.0
        self._phase  = 0.0
        self._pd     = 1
        self._bars   = [0.0] * 20
        self._flash  = 0

        self.root   = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack()
        self.root.title("J.A.R.V.I.S  —  AI Mode")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._register()
        self._tick()

    def _tick(self):
        self._update_bars()
        self._draw()
        self._angle  = (self._angle + (3.5 if self._active else 0.8)) % 360
        self._phase += 0.07 * self._pd
        if   self._phase > math.pi: self._pd = -1
        elif self._phase < 0:       self._pd =  1
        if self._flash > 0:
            self._flash -= 1
        self.root.after(35, self._tick)

    def _update_bars(self):
        for i in range(len(self._bars)):
            t = random.uniform(0.15, 1.0) if self._active else 0.04
            self._bars[i] += (t - self._bars[i]) * 0.35

    def _draw(self):
        cv = self.canvas
        cx = cy = self.W // 2
        cv.delete("all")

        # Grid
        gc = "#060e16"
        for x in range(0, self.W, 40):
            cv.create_line(x, 0, x, self.H, fill=gc)
        for y in range(0, self.H, 40):
            cv.create_line(0, y, self.W, y, fill=gc)

        # Corner brackets
        bc = self._color if self._active else CDIM
        for bx, by, sx, sy in [(22,22,1,1),(self.W-22,22,-1,1),
                                (22,self.H-22,1,-1),(self.W-22,self.H-22,-1,-1)]:
            cv.create_line(bx, by, bx+40*sx, by, fill=bc, width=2)
            cv.create_line(bx, by, bx, by+40*sy, fill=bc, width=2)

        # Outer glow rings
        for r, a in [(225,0.07),(205,0.13),(188,0.19)]:
            cv.create_oval(cx-r,cy-r,cx+r,cy+r,
                           outline=self._dim(self._color,a), width=1, fill="")

        # Main rotating arc
        R = 172
        if self._active:
            start  = int(self._angle) % 360
            extent = 210 + int(50*math.sin(self._phase))
            for i, (w, a) in enumerate([(14,1.0),(8,0.55),(3,0.25)]):
                off = i*2
                cv.create_arc(cx-R+off,cy-R+off,cx+R-off,cy+R-off,
                              start=start, extent=extent,
                              style=tk.ARC,
                              outline=self._dim(self._color,a), width=w)
            cv.create_arc(cx-R,cy-R,cx+R,cy+R,
                          start=(-self._angle*1.5)%360, extent=55,
                          style=tk.ARC,
                          outline=self._dim(AMBER,0.55), width=4)
        else:
            cv.create_oval(cx-R,cy-R,cx+R,cy+R, outline=CDIM, width=2, fill="")

        # Tick marks
        tr = R+20
        for i in range(24):
            a   = math.radians(i*15+self._angle*0.3)
            x1  = cx+tr*math.cos(a); y1 = cy+tr*math.sin(a)
            tr2 = tr+(9 if i%6==0 else 4)
            x2  = cx+tr2*math.cos(a); y2 = cy+tr2*math.sin(a)
            col = self._color if (i%6==0 and self._active) else CDIM
            cv.create_line(x1,y1,x2,y2, fill=col, width=(2 if i%6==0 else 1))

        # Inner hexagon
        hr  = 82
        pts = []
        for i in range(6):
            a = math.radians(i*60+30)
            pts += [cx+hr*math.cos(a), cy+hr*math.sin(a)]
        cv.create_polygon(pts, outline=self._dim(self._color,0.45 if self._active else 0.18),
                          fill=DARK, width=2)
        for i in range(6):
            a = math.radians(i*60+30)
            cv.create_line(cx+20*math.cos(a),cy+20*math.sin(a),
                           cx+hr*math.cos(a), cy+hr*math.sin(a),
                           fill=self._dim(self._color,0.22), width=1)

        # Centre orb
        orb_r = 34+(int(8*math.sin(self._phase)) if self._active else 0)
        orb_c = self._color if self._active else DARK
        for gr, ga in [(orb_r+24,0.07),(orb_r+15,0.14),(orb_r+7,0.28)]:
            cv.create_oval(cx-gr,cy-gr,cx+gr,cy+gr,
                           outline=self._dim(self._color,ga), width=4, fill="")
        cv.create_oval(cx-orb_r,cy-orb_r,cx+orb_r,cy+orb_r,
                       fill=orb_c, outline=self._color, width=2)
        for ir in [18,10,5]:
            cv.create_oval(cx-ir,cy-ir,cx+ir,cy+ir,
                           outline=self._dim(self._color,0.6), width=1, fill="")
        cv.create_text(cx,cy, text="J", fill=BG if self._active else CDIM,
                       font=("Courier New",22,"bold"))

        # Wake flash
        if self._flash > 0:
            fc = self._dim(self._color, min(0.35, self._flash/12.0))
            cv.create_rectangle(0,0,self.W,self.H, fill=fc, outline="")

        # Waveform
        n=len(self._bars); bw=8; bgap=4
        total=n*(bw+bgap)-bgap; bx0=cx-total//2
        by_ctr=cy+265; bmax=55
        for i, h in enumerate(self._bars):
            bh=int(h*bmax); x=bx0+i*(bw+bgap)
            col=self._dim(self._color,0.8 if h>0.5 else 0.32)
            cv.create_rectangle(x,by_ctr-bh,x+bw,by_ctr+bh, fill=col, outline="")

        # Status
        st, sc = STATUS_TEXT.get(self._status, STATUS_TEXT["standby"])
        cv.create_text(cx,cy+215, text=st, fill=sc, font=("Courier New",12,"bold"))

        # Title
        cv.create_text(cx,38, text="J.A.R.V.I.S",
                       fill=self._dim(self._color,0.9), font=("Courier New",20,"bold"))
        cv.create_text(cx,62, text="AI-POWERED  —  NO HARDCODED COMMANDS",
                       fill=CDIM, font=("Courier New",7))

        # AI status badge
        ex = self.brain.executor
        if ex.ai_available:
            # Colour by backend quality
            if ex._backend == "groq":
                ai_col  = GREEN
                total   = len(ex._groq_keys)
                cur     = ex._groq_idx + 1
                exhaust = sum(1 for v in ex._groq_exhausted.values() if v)
            elif ex._backend == "gemini":
                ai_col  = CYAN
                total   = len(ex._gemini_keys)
                cur     = ex._gemini_idx + 1
                exhaust = sum(1 for v in ex._gemini_exhausted.values() if v)
            else:
                ai_col  = AMBER
                total   = len(ex._openai_keys)
                cur     = ex._openai_idx + 1
                exhaust = sum(1 for v in ex._openai_exhausted.values() if v)
            key_txt = f"KEY {cur}/{total}" + (f"  ({exhaust} EXHAUSTED)" if exhaust else "")
            ai_txt  = f"AI ONLINE  —  {ex.backend_name.upper()}  —  {key_txt}"
        else:
            ai_col = RED
            ai_txt = "AI OFFLINE — SET GROQ_API_KEY (FREE) AT CONSOLE.GROQ.COM"
        cv.create_text(cx, 82, text=ai_txt, fill=ai_col, font=("Courier New", 8, "bold"))

        # Face auth status
        if self._face_auth and self._face_auth.available:
            if self._face_auth.is_registered:
                fc_col = GREEN
                fc_txt = "FACE ID  ●  OWNER VERIFIED"
            else:
                fc_col = AMBER
                fc_txt = "FACE ID  ○  NOT REGISTERED — SAY: JARVIS REGISTER MY FACE"
        else:
            fc_col = self._dim(CDIM, 0.5)
            fc_txt = "FACE ID  ○  INSTALL face-recognition TO ENABLE"
        cv.create_text(cx, 100, text=fc_txt, fill=fc_col,
                       font=("Courier New", 8))

        # Bottom
        now = datetime.datetime.now()
        cv.create_text(cx,self.H-30,
                       text=f"{now:%H:%M:%S}   |   VOICE MODE   |   WINDOWS",
                       fill=CDIM, font=("Courier New",8))
        cv.create_text(cx,self.H-14,
                       text="ALL ANSWERS DELIVERED BY VOICE  —  AI UNDERSTANDS ANYTHING",
                       fill=self._dim(CDIM,0.55), font=("Courier New",7))

    @staticmethod
    def _dim(h: str, t: float) -> str:
        def p(x): return [int(x.lstrip("#")[i:i+2],16) for i in (0,2,4)]
        bg, fg = p(BG), p(h)
        r = [int(bg[i]+(fg[i]-bg[i])*t) for i in range(3)]
        return "#{:02x}{:02x}{:02x}".format(*r)

    def _register(self):
        self.brain.on_status = self._cb_status
        self.brain.on_wake   = self._cb_wake

    def _cb_status(self, status):
        self.root.after(0, lambda: self._apply(status))

    def _cb_wake(self):
        self.root.after(0, self._do_flash)

    def _apply(self, status):
        self._status = status
        self._color, self._active = STATUS_COLOR.get(status, (CDIM, False))

    def _do_flash(self):
        self._flash = 12

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self):
        self.brain.stop()
        self.root.destroy()