#!/usr/bin/env python3
"""
MJPEG 流服务 + 网页控制面板。截图/录像全部在浏览器端保存。
同网段访问 http://<机子IP>:5000

启动:
  python3 web_viewer.py          # camera_server 需已运行
"""

import cv2
import os
import logging
import threading
import time
import yaml
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

logging.basicConfig(level=logging.INFO, format="[web_viewer] %(message)s")
log = logging.getLogger("web_viewer")

_jpeg_frame = None
_lock = threading.Lock()
_running = True


class ThreadedServer(ThreadingMixIn, HTTPServer):
    pass


_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAP 相机</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;align-items:center}
h1{font-size:1.2rem;margin:12px 0 6px;opacity:.8;text-align:center}
#box{position:relative;max-width:100vw;max-height:70vh;background:#000;border-radius:6px;overflow:hidden;text-align:center}
#stream{max-width:100vw;max-height:70vh;object-fit:contain}
#recBadge{display:none;position:absolute;top:10px;left:12px;background:#dc3545;color:#fff;padding:4px 12px;border-radius:4px}
#recBadge.on{display:block}
.controls{text-align:center;margin:12px 0;gap:12px}
.btn{padding:10px 24px;border:none;border-radius:6px;font-size:1rem;cursor:pointer}
.btn-capture{background:#ffc107;color:#000}
.btn-record{background:#dc3545;color:#fff}
.btn-record.on{background:#198754}
#status{text-align:center;font-size:.85rem;opacity:.6;margin:6px 0 14px}
</style>
</head>
<body>
<h1>CAP 相机</h1>
<div id="box" style="position:relative"><img id="stream" src="/stream"/><div id="recBadge">● 录制中</div><div id="reconnBanner" style="display:none;position:absolute;inset:0;background:rgba(0,0,0,.75);color:#fff;z-index:10;display:none;flex-direction:column;align-items:center;justify-content:center;gap:8px;font-size:1.2rem"><span>⏳ 连接断开</span><span id="reconnMsg" style="font-size:.9rem;opacity:.8">正在重连 (1/2)...</span></div></div>
<div class="controls">
<button class="btn btn-capture" onclick="doCapture()">截图</button>
<button class="btn btn-record" id="recBtn" onclick="doRecord()">录像</button>
</div>
<div id="status">等待画面...</div>
<canvas id="cv" style="display:none"></canvas>
<script>
const img=document.getElementById("stream"),cv=document.getElementById("cv"),ctx=cv.getContext("2d"),st=document.getElementById("status"),bd=document.getElementById("recBadge"),rb=document.getElementById("recBtn");
var rec=null,chunks=[],fid=null;
function cap(){if(img.complete&&img.naturalWidth){cv.width=img.naturalWidth;cv.height=img.naturalHeight;ctx.drawImage(img,0,0)}}
function dl(b,n){var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=n;a.click();URL.revokeObjectURL(a.href)}
function ts(){var d=new Date();return d.getFullYear()+("0"+(d.getMonth()+1)).slice(-2)+("0"+d.getDate()).slice(-2)+"_"+("0"+d.getHours()).slice(-2)+("0"+d.getMinutes()).slice(-2)+("0"+d.getSeconds()).slice(-2)}
function doCapture(){cap();cv.toBlob(function(b){if(b)dl(b,"capture_"+ts()+".jpg")},"image/jpeg",0.92)}
function doRecord(){
if(rec&&rec.state==="recording"){rec.stop();rb.textContent="录像";rb.classList.remove("on");bd.classList.remove("on");if(fid){cancelAnimationFrame(fid);fid=null}return}
cap();chunks=[];
try{rec=new MediaRecorder(cv.captureStream(30),{mimeType:"video/webm"})}catch(e){st.textContent="浏览器不支持录像";return}
rec.ondataavailable=function(e){if(e.data.size)chunks.push(e.data)};
rec.onstop=function(){var b=new Blob(chunks,{type:"video/webm"});if(b.size>0)dl(b,"record_"+ts()+".webm");chunks=[]};
rec.start(1000);rb.textContent="停止";rb.classList.add("on");bd.classList.add("on");
function f(){if(rec&&rec.state==="recording"){cap();fid=requestAnimationFrame(f)}}f()}
img.onload=function(){st.textContent="实时 "+img.naturalWidth+"x"+img.naturalHeight};
img.onerror=function(){st.textContent="等待画面..."};
setInterval(function(){if(st.textContent.includes("等待")&&img.complete&&img.naturalWidth>0)st.textContent="实时 "+img.naturalWidth+"x"+img.naturalHeight},2000);
var _ra=0,_rm=2,_rb=document.getElementById("reconnBanner"),_rmsg=document.getElementById("reconnMsg");
img.onerror=function(){_ra++;if(_ra<=_rm){_rb.style.display="flex";_rmsg.textContent="正在重连 ("+_ra+"/"+_rm+")...";st.textContent="";setTimeout(function(){img.src="/stream?"+Date.now()},800)}else{_rmsg.textContent="连接已断开，请检查服务器";st.textContent="离线"}};
img.onload=function(){if(_ra>0){_rb.style.display="none";_ra=0;st.textContent="重连成功";setTimeout(function(){st.textContent="实时 "+img.naturalWidth+"x"+img.naturalHeight},2000)}};
</script>
</body></html>"""


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML.encode("utf-8"))
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            while _running:
                with _lock:
                    data = _jpeg_frame
                if data is None:
                    time.sleep(0.02)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                time.sleep(0.03)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        log.info(f"{self.client_address[0]} {fmt % args}")


def capture_loop():
    global _jpeg_frame
    camera_id = 2
    cfg_path = Path("./config/capture.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            camera_id = yaml.safe_load(f).get("v4l2", {}).get("virtualDeviceId", 2)
    dev_path = f"/dev/video{camera_id}"
    cap = None
    while _running:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap = None
        log.warning(f"无法打开 {dev_path}，10 秒后重试...")
        for _ in range(100):
            if not _running:
                return
            time.sleep(0.1)
    log.info(f"已连接 {dev_path}")
    _fail = 0
    while _running:
        ret, frame = cap.read()
        if not ret:
            _fail += 1
            if _fail >= 3:
                log.error("相机已断开，退出进程")
                os._exit(1)
            log.warning(f"读帧失败 ({_fail}/3)，重试中...")
            time.sleep(1)
            continue
        _fail = 0
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        with _lock:
            _jpeg_frame = jpeg.tobytes()
    cap.release()
    log.info("采集已停止")


def main():
    global _running
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()
    port = 5000
    server = ThreadedServer(("0.0.0.0", port), StreamHandler)
    log.info("MJPEG 流 + 控制面板")
    log.info(f"   http://<本机IP>:{port}")
    log.info("   Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        server.shutdown()
        log.info("已停止")


if __name__ == "__main__":
    main()

