from flask import Flask, request, jsonify, send_file, abort
import numpy as np
import cv2
import base64
import os

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_file('index.html')

# avoid FileNotFoundError when browser requests favicon
@app.route('/favicon.ico')
def favicon():
    # no favicon available — return 204 No Content so browser stops requesting
    return ('', 204)

@app.route('/api/process', methods=['POST'])
def api_process():
    f = request.files.get('file')
    mode = request.form.get('mode', 'gray8')
    if not f:
        return jsonify({'error':'missing file'}), 400
    try:
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        params = {}

        # If PCA mode: decode color and work on luminance (Y) channel then recompose color
        if mode == 'pca_nn':
            # PCA+NearestNeighbor inference (DB file: models/pca_db.npz)
            db_path = os.path.join('models', 'pca_db.npz')
            if not os.path.isfile(db_path):
                return jsonify({'error': 'missing PCA DB: models/pca_db.npz'}), 400
            db = np.load(db_path, allow_pickle=True)
            mu = db['mu']            # shape (n_pixels,)
            comps = db['components'] # shape (k, n_pixels)
            feats = db['features']   # shape (N, k)
            labels = db['labels']    # shape (N,) (string)
            thresh = float(db.get('threshold', 0.20))

            # decode grayscale (use robust decoder to match training preprocessing)
            gray = ensure_gray_uint8_from_bytes(data)
            H, W = gray.shape
            x = gray.astype(np.float32).ravel()
            x = x / 255.0  # optional normalization if DB used it

            # center and project: p = comps @ (x - mu)
            v = x - mu
            p = comps.dot(v)   # shape (k,)

            # nearest neighbor in feature space
            dists = np.linalg.norm(feats - p.reshape(1, -1), axis=1)
            best = int(np.argmin(dists))
            best_dist = float(dists[best])
            best_label = str(labels[best])

            # reconstruct and compute relative reconstruction error
            x_hat = mu + comps.T.dot(p)
            rec_err = float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-12))

            # decide unknown if reconstruction error > threshold OR distance too large
            unknown = rec_err > thresh or best_dist > (np.mean(dists) + 3*np.std(dists))

            # prepare annotated preview (draw label or UNKNOWN)
            # use original color if available
            try:
                color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if color is None:
                    color = cv2.cvtColor((gray).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            except Exception:
                color = cv2.cvtColor((gray).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            text = f"{best_label} ({best_dist:.3f})" if not unknown else f"UNKNOWN ({rec_err:.3f})"
            cv2.putText(color, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if not unknown else (0,0,255), 2)

            params = {'label': best_label, 'distance': best_dist, 'reconstruction_error': rec_err, 'unknown': bool(unknown)}
            return jsonify({'dataURL': to_data_url_png(color), 'message': 'pca_nn_ok', 'params': params}), 200

        if mode == 'detect':
            # Object detection using OpenCV DNN (MobileNet-SSD example).
            # Place MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel in ./models/
            proto = os.path.join('models', 'MobileNetSSD_deploy.prototxt')
            model_file = os.path.join('models', 'MobileNetSSD_deploy.caffemodel')
            if not os.path.isfile(proto) or not os.path.isfile(model_file):
                return jsonify({'error': 'missing detection model files in ./models/'}), 400
            img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_color is None:
                return jsonify({'error':'failed to decode color image for detection'}), 400

            net = cv2.dnn.readNetFromCaffe(proto, model_file)
            (h, w) = img_color.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img_color, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
                       "cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
            out_list = []
            for i in range(detections.shape[2]):
                score = float(detections[0, 0, i, 2])
                if score < 0.4:
                    continue
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int").tolist()
                label = CLASSES[idx] if idx < len(CLASSES) else f'class_{idx}'
                out_list.append({'label': label, 'score': round(score,3), 'box': [int(startX), int(startY), int(endX), int(endY)]})
                # draw box + label on image
                cv2.rectangle(img_color, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{label}: {int(score*100)}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img_color, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            img = img_color
            params = {'detections': out_list, 'count': len(out_list)}
            data_url = to_data_url_png(img)
            return jsonify({'dataURL': data_url, 'message':'detect_ok', 'params': params}), 200

        # NOTE: 'pca' (legacy PCA denoising pipeline) removed to simplify server.
        # If client still sends mode='pca' return a clear error and point to supported modes.
        elif mode == 'pca':
            return jsonify({
                'error': "mode 'pca' has been removed on server. Use 'pca_nn' for PCA+NN inference or 'detect' for object detection."
            }), 400

        else:
            # other modes expect grayscale input
            img = ensure_gray_uint8_from_bytes(data)

            if mode in ('gamma', 'powerlaw', 'power-law'):
                gamma_param = request.form.get('gamma') or request.form.get('gamma_value') or request.form.get('g')
                gamma_val = float(gamma_param) if gamma_param is not None else None

                # histogram PMF + CDF and percentiles
                flat = img.flatten().astype(np.uint8)
                h, bins = np.histogram(flat, bins=256, range=(0,256))
                pmf = h.astype(np.float64) / (h.sum() + 1e-12)
                cdf = pmf.cumsum()
                p10 = float(np.percentile(flat, 10))
                p30 = float(np.percentile(flat, 30))
                p50 = float(np.percentile(flat, 50))
                p70 = float(np.percentile(flat, 70))
                p90 = float(np.percentile(flat, 90))
                spread = p90 - p10

                def cdf_percentile_to_level(pct):
                    return float(np.interp(pct, cdf, bins[:-1]))

                r_med = cdf_percentile_to_level(0.5)
                r_dark = cdf_percentile_to_level(0.3)
                r_bright = cdf_percentile_to_level(0.7)

                if gamma_val is None:
                    target = 0.5
                    def safe_gamma_from_mapping(r_level, tgt=target):
                        r_norm = max(r_level / 255.0, 1e-6)
                        try:
                            return float(np.log(tgt) / np.log(r_norm))
                        except Exception:
                            return 1.0

                    gamma_med = safe_gamma_from_mapping(r_med, target)

                    if spread < 25:
                        if p50 < 110:
                            gamma_spread = 0.7
                        elif p50 > 145:
                            gamma_spread = 1.4
                        else:
                            gamma_spread = 0.95
                    else:
                        skew_corr = (128.0 - p50) / 128.0
                        gamma_spread = 1.0 + skew_corr * 0.6

                    gamma_est = 0.75 * gamma_med + 0.25 * gamma_spread
                    gamma_val = float(np.clip(gamma_est, 0.4, 3.0))

                r = img.astype(np.float32) / 255.0
                s = np.power(r, gamma_val)
                img = np.clip((s * 255.0), 0, 255).astype(np.uint8)
                params = {'gamma': float(gamma_val), 'p10': p10, 'p50': p50, 'p90': p90, 'spread': spread}

            elif mode == 'negative':
                img = 255 - img

            elif mode == 'threshold':
                T = int(request.form.get('threshold') or request.form.get('T') or 128)
                _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

            elif mode in ('log', 'logarithmic', 'logrithmic'):
                c_val = float(request.form.get('c', '2.0'))
                r = img.astype(np.float32) / 255.0
                s = np.log1p(c_val * r)
                s /= (s.max() + 1e-12)
                img = (s * 255.0).astype(np.uint8)

            elif mode == 'median':
                # Enhanced median pipeline for salt-and-pepper:
                # optional pre-gaussian 'gauss', median kernel 'k', morphological kernel 'morph',
                # and number of median passes 'passes'.
                try:
                    k = int(request.form.get('k', '3'))
                except Exception:
                    k = 3
                if k % 2 == 0:
                    k += 1
                k = max(3, min(31, k))

                try:
                    g = int(request.form.get('gauss', '0'))
                except Exception:
                    g = 0
                if g % 2 == 0 and g > 0:
                    g += 1
                if g < 1:
                    g = 0

                try:
                    morph = int(request.form.get('morph', '0'))
                except Exception:
                    morph = 0
                if morph % 2 == 0 and morph > 0:
                    morph += 1
                if morph < 1:
                    morph = 0

                try:
                    passes = int(request.form.get('passes', '1'))
                except Exception:
                    passes = 1
                passes = max(1, min(4, passes))

                # pipeline
                if g > 0:
                    img = cv2.GaussianBlur(img, (g, g), 0)
                for _ in range(passes):
                    img = cv2.medianBlur(img, k)
                if morph > 0:
                    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

                # optional final denoise fallback (light)
                if request.form.get('nlm', '0') == '1':
                    # h parameter small so details preserved
                    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

                params = {'median_k': int(k), 'pre_gauss': int(g), 'morph_k': int(morph), 'passes': int(passes)}

            elif mode in ('piecewise', 'piecewise_linear', 'stretch'):
                flat = img.flatten().astype(np.uint8)
                h, bins = np.histogram(flat, bins=256, range=(0,256))
                pmf = h.astype(np.float64) / (h.sum() + 1e-12)
                win = 7
                sigma = 2.0
                xw = np.arange(-win, win+1)
                kernel = np.exp(-0.5*(xw/sigma)**2)
                kernel = kernel / kernel.sum()
                pmf_smooth = np.convolve(pmf, kernel, mode='same')

                peaks = []
                thresh = pmf_smooth.max() * 0.03
                for i in range(1, len(pmf_smooth)-1):
                    if pmf_smooth[i] > pmf_smooth[i-1] and pmf_smooth[i] > pmf_smooth[i+1] and pmf_smooth[i] >= thresh:
                        peaks.append((pmf_smooth[i], i))
                peaks.sort(reverse=True)

                if len(peaks) >= 2:
                    cand_idx = [p[1] for p in peaks[:6]]
                    best_pair = (0, (cand_idx[0], cand_idx[0]))
                    for i in range(len(cand_idx)):
                        for j in range(i+1, len(cand_idx)):
                            d = abs(cand_idx[i] - cand_idx[j])
                            if d > best_pair[0]:
                                best_pair = (d, (cand_idx[i], cand_idx[j]))
                    r1_level = int(np.clip(round(min(best_pair[1])), 1, 254))
                    r2_level = int(np.clip(round(max(best_pair[1])), r1_level+1, 255))
                else:
                    p30 = float(np.percentile(flat, 30))
                    p70 = float(np.percentile(flat, 70))
                    r1_level = int(np.clip(round(p30), 1, 254))
                    r2_level = int(np.clip(round(p70), r1_level+1, 255))

                if 'auto' in request.form:
                    r1 = r1_level
                    r2 = r2_level
                    s1 = int(np.clip(round(r1_level * 0.6), 0, 255))
                    s2 = int(np.clip(round(r2_level * 1.05), 0, 255))
                else:
                    r1 = int(request.form.get('r1', r1_level))
                    s1 = int(request.form.get('s1', int(np.clip(round(r1_level * 0.6),0,255))))
                    r2 = int(request.form.get('r2', r2_level))
                    s2 = int(request.form.get('s2', int(np.clip(round(r2_level * 1.05),0,255))))

                r1 = max(1, min(254, r1))
                r2 = max(r1+1, min(255, r2))
                s1 = int(np.clip(s1, 0, 255))
                s2 = int(np.clip(s2, 0, 255))

                x = img.astype(np.float32)
                y = np.zeros_like(x, dtype=np.float32)
                idx1 = x <= r1
                y[idx1] = (s1 / float(r1)) * x[idx1]
                idx2 = (x > r1) & (x <= r2)
                y[idx2] = s1 + (s2 - s1) * (x[idx2] - r1) / float(r2 - r1)
                idx3 = x > r2
                y[idx3] = s2 + (255.0 - s2) * (x[idx3] - r2) / float(255 - r2)
                img = np.clip(np.round(y), 0, 255).astype(np.uint8)

        data_url = to_data_url_png(img)
        return jsonify({'dataURL': data_url, 'message':'ok', 'params': params}), 200
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# simple static file proxy so browser can fetch src/styles.css, src/app.js, etc.
@app.route('/<path:filename>')
def static_files(filename):
    # send only if file exists in project root; otherwise return 404
    safe_path = os.path.abspath(filename)
    project_root = os.path.abspath('.')
    # basic safety: ensure requested file path is under project root
    if not safe_path.startswith(project_root):
        return abort(404)
    if not os.path.isfile(safe_path):
        # file missing — return 404 (avoid raising FileNotFoundError)
        return abort(404)
    return send_file(safe_path)

def ensure_gray_uint8_from_bytes(img_bytes):
    """
    Decode bytes robustly and return a 2D uint8 grayscale image.
    Tries several imdecode flags and handles RGBA / 16-bit / float cases.
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    last_err = None
    for flag in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYDEPTH, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR):
        try:
            img = cv2.imdecode(arr, flag)
        except Exception as e:
            img = None
            last_err = e
        if img is None:
            continue

        # If has alpha channel, drop/convert alpha
        if img.ndim == 3 and img.shape[2] == 4:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception:
                img = img[:, :, :3]

        # If color, convert to grayscale
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize/cast non-uint8 types
        if img.dtype != np.uint8:
            if img.dtype == np.uint16:
                # scale 16-bit -> 8-bit
                img = (img >> 8).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Final check: must be 2D uint8
        if img is not None and img.ndim == 2 and img.dtype == np.uint8:
            return img

    raise RuntimeError("Cannot decode image to grayscale 8-bit. " + (str(last_err) if last_err is not None else "unknown error"))

def to_data_url_png(img):
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf).decode('ascii')
    return f"data:image/png;base64,{b64}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)