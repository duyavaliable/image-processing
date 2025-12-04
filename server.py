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


        if mode == 'count_objects':
            # Count specific object class using YOLO (requires models/yolov5s.onnx)
            yolo_onnx = os.path.join('models', 'yolov5s.onnx')
            if not os.path.isfile(yolo_onnx):
                return jsonify({'error': 'Missing YOLO ONNX: models/yolov5s.onnx'}), 400
            
            img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_color is None:
                return jsonify({'error':'failed to decode color image for counting'}), 400
            
            h0, w0 = img_color.shape[:2]
            inp_size = int(request.form.get('inp_size', '640'))
            blob = cv2.dnn.blobFromImage(img_color, 1/255.0, (inp_size, inp_size), [0,0,0], swapRB=True, crop=False)
            net = cv2.dnn.readNetFromONNX(yolo_onnx)
            net.setInput(blob)
            preds = net.forward()
            if preds.ndim == 3:
                preds = preds[0]
            
            boxes, confidences, class_ids = [], [], []
            conf_thresh = float(request.form.get('conf_thresh', 0.25))
            iou_thresh = float(request.form.get('iou_thresh', 0.45))
            
            for row in preds:
                obj_conf = float(row[4])
                if obj_conf <= 0.01:
                    continue
                class_scores = row[5:]
                class_id = int(class_scores.argmax())
                cls_conf = float(class_scores[class_id])
                conf = obj_conf * cls_conf
                if conf < conf_thresh:
                    continue
                x_c, y_c, ww, hh = row[0:4]
                x = (x_c - ww/2) * w0 / inp_size
                y = (y_c - hh/2) * h0 / inp_size
                w = ww * w0 / inp_size
                h = hh * h0 / inp_size
                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(int(class_id))
            
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
            
            CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
                       "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
                       "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
                       "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
                       "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                       "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
                       "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
                       "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
                       "hair drier","toothbrush"]
            
            # Get target class from form (default 'all')
            target_class = request.form.get('target_class', 'all').strip().lower()
            print(f"[count_objects] target_class received: '{target_class}'")  # DEBUG
            
            count = 0
            for i in (indices.flatten() if len(indices) else []):
                x,y,w,h = boxes[i]
                cls = class_ids[i] if i < len(class_ids) else -1
                label = CLASSES[cls] if (isinstance(CLASSES, list) and cls < len(CLASSES)) else f'class_{cls}'
                score = confidences[i]
                
                # Filter by target_class
                # FIXED: ensure exact match (strip whitespace from both)
                label_normalized = label.strip().lower()
                target_normalized = target_class.strip().lower()
                if target_normalized == 'all' or label_normalized == target_normalized:
                    count += 1
                    cv2.rectangle(img_color, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(img_color, f"{label} {score:.2f}", (x, max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                else:
                    # DEBUG: log skipped detections
                    print(f"[count_objects] skipped: label='{label}' (normalized='{label_normalized}') vs target='{target_normalized}'")
            
            # Draw count overlay
            text = f"Total: {count}" if target_class == 'all' else f"{target_class.capitalize()}: {count}"
            cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            
            params = {'count': count, 'target_class': target_class, 'backend':'yolo_count'}
            return jsonify({'dataURL': to_data_url_png(img_color), 'message': f'Counted {count} {target_class}', 'params': params}), 200

        if mode == 'denoise':
            # Auto noise detection + removal pipeline
            img = ensure_gray_uint8_from_bytes(data)
            
            # Step 1: Detect noise type
            noise_type, noise_params = detect_noise_type(img)
            # DEBUG logs: raw detection info and types
            print(f"[denoise] detected noise_type={noise_type}")
            try:
                print(f"[denoise] raw noise_params: {repr(noise_params)}")
                for k, v in (noise_params.items() if isinstance(noise_params, dict) else []):
                    try:
                        print(f"[denoise] noise_params[{k}] type={type(v)} preview={str(v)[:200]}")
                    except Exception:
                        print(f"[denoise] noise_params[{k}] type={type(v)} (preview failed)")
            except Exception:
                print("[denoise] failed to print noise_params details")
            
            # Step 2: Apply filter based on noise type
            if noise_type == 'salt_pepper':
                img_denoised = adaptive_median_filter(img, max_size=7)
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                img_denoised = cv2.morphologyEx(img_denoised, cv2.MORPH_OPEN, se)
                method = 'Adaptive Median + Opening'
            
            elif noise_type == 'gaussian':
                sigma = max(10, min(100, int(np.sqrt(noise_params.get('variance', 100)))))
                img_denoised = cv2.bilateralFilter(img, 9, sigma, sigma)
                method = 'Bilateral Filter'
            
            elif noise_type == 'periodic':
                try:
                    img_denoised, freqs = notch_filter_auto(img)
                    method = 'Notch Filter'
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[denoise] notch_filter_auto failed: {e!r} — falling back to bilateral")
                    img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
                    freqs = []
                    method = 'Notch Filter (failed->bilateral)'
            
            elif noise_type == 'speckle':
                img_denoised = lee_filter(img, window_size=5)
                method = 'Lee Filter'
            
            else:
                img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
                method = 'Bilateral Filter (fallback)'
            
            # DEBUG: check img_denoised before encoding
            print(f"[denoise] img_denoised dtype={img_denoised.dtype}, shape={img_denoised.shape}, sample={img_denoised[0,0] if img_denoised.size > 0 else 'empty'}")
            
            # Convert noise_params to JSON-safe structure and log
            try:
                noise_params_safe = numpy_to_python(noise_params)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[denoise] numpy_to_python failed: {e!r}; using repr fallback")
                noise_params_safe = {'_error': str(e), 'raw': repr(noise_params)}
            print(f"[denoise] noise_params_safe: {repr(noise_params_safe)[:1000]}")

            # Encode to PNG dataURL
            try:
                dataURL = to_data_url_png(img_denoised)
                print(f"[denoise] to_data_url_png returned dataURL length={len(dataURL) if dataURL else 0}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[denoise] to_data_url_png failed: {e!r}")
                dataURL = None
            
            params = {
                'noise_type': noise_type,
                'method': method,
                'noise_params': noise_params_safe
            }
            return jsonify({'dataURL': dataURL, 'message': f'Denoised: {noise_type}', 'params': params}), 200

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

def detect_noise_type(img):
    """
    Detect noise type: salt_pepper, gaussian, periodic, speckle
    Returns: (noise_type: str, params: dict)
    """
    H, W = img.shape
    
    # 1. Check salt-and-pepper (>1% extreme pixels)
    n_black = int(np.sum(img == 0))
    n_white = int(np.sum(img == 255))
    extreme_ratio = float((n_black + n_white) / (H * W))
    
    if extreme_ratio > 0.01:
        return ('salt_pepper', {'extreme_ratio': extreme_ratio, 'black': n_black, 'white': n_white})
    
    # 2. Check periodic noise (frequency domain peaks)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag_log = np.log1p(magnitude)
    
    cy, cx = H // 2, W // 2
    mask = np.ones((H, W), dtype=bool)
    mask[cy-10:cy+10, cx-10:cx+10] = False  # exclude DC
    
    threshold = mag_log[mask].mean() + 3 * mag_log[mask].std()
    peaks = np.where((mag_log > threshold) & mask)
    
    if len(peaks[0]) > 5:
        peak_coords = [(int(peaks[0][i]), int(peaks[1][i])) for i in range(min(5, len(peaks[0])))]
        return ('periodic', {'num_peaks': len(peaks[0]), 'peak_coords': peak_coords})
    
    # 3. Check Gaussian (moderate variance, no extreme pixels)
    flat = img.flatten().astype(np.float32)
    mean_val = float(np.mean(flat))
    var = float(np.var(flat))
    
    if 100 < var < 2000:
        return ('gaussian', {'mean': mean_val, 'variance': var})
    
    # 4. Check speckle (high coefficient of variation)
    cv_val = float(np.std(flat) / (mean_val + 1e-12))
    if cv_val > 0.3:
        return ('speckle', {'cv': cv_val, 'variance': var})
    
    # fallback
    return ('unknown', {'variance': var})

def adaptive_median_filter(img, max_size=7):
    """
    Adaptive median filter for salt-and-pepper noise
    """
    H, W = img.shape
    out = img.copy()
    
    def get_median(window):
        return int(np.median(window))
    
    for y in range(H):
        for x in range(W):
            size = 3
            while size <= max_size:
                k = size // 2
                y1, y2 = max(0, y-k), min(H, y+k+1)
                x1, x2 = max(0, x-k), min(W, x+k+1)
                window = img[y1:y2, x1:x2]
                
                z_min = int(window.min())
                z_max = int(window.max())
                z_med = get_median(window)
                z_xy = int(img[y, x])
                
                # Stage A
                if z_min < z_med < z_max:
                    # Stage B
                    if z_min < z_xy < z_max:
                        out[y, x] = z_xy
                    else:
                        out[y, x] = z_med
                    break
                else:
                    size += 2
                    if size > max_size:
                        out[y, x] = z_med
    
    return out

def notch_filter_auto(img):
    """Notch filter: remove periodic noise peaks in frequency domain"""
    H, W = img.shape
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag_log = np.log1p(magnitude)
    
    cy, cx = H // 2, W // 2
    mask = np.ones((H, W), dtype=bool)
    mask[cy-10:cy+10, cx-10:cx+10] = False
    
    threshold = mag_log[mask].mean() + 3 * mag_log[mask].std()
    peaks = np.where((mag_log > threshold) & mask)
    
    print(f"[notch_filter_auto] found {len(peaks[0])} peaks")
    
    # Create notch mask (circular rejection)
    notch_mask = np.ones((H, W, 2), dtype=np.float32)
    for py, px in zip(peaks[0][:10], peaks[1][:10]):
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - px)**2 + (yy - py)**2)
        notch_mask[dist < 5] = 0
    
    dft_shift_filtered = dft_shift * notch_mask
    dft_ishift = np.fft.ifftshift(dft_shift_filtered)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # CRITICAL: normalize to [0,255] uint8
    # img_back might have values outside [0,255] after magnitude
    img_min = img_back.min()
    img_max = img_back.max()
    print(f"[notch_filter_auto] img_back range: [{img_min}, {img_max}]")
    
    if img_max > img_min:
        img_back = ((img_back - img_min) / (img_max - img_min) * 255.0)
    else:
        img_back = np.zeros_like(img_back)
    
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    print(f"[notch_filter_auto] output dtype={img_back.dtype}, shape={img_back.shape}, sample={img_back[0,0]}")
    
    peak_freqs = [(int(py), int(px)) for py, px in zip(peaks[0][:5], peaks[1][:5])]
    return img_back, peak_freqs

def lee_filter(img, window_size=5):
    """
    Lee filter for speckle noise (multiplicative noise)
    """
    H, W = img.shape
    out = img.copy().astype(np.float32)
    k = window_size // 2
    
    for y in range(k, H - k):
        for x in range(k, W - k):
            window = img[y-k:y+k+1, x-k:x+k+1].astype(np.float32)
            mean_w = window.mean()
            var_w = window.var()
            
            # Lee filter formula
            if var_w > 0:
                k_lee = var_w / (var_w + mean_w**2 / (mean_w + 1e-12))
                out[y, x] = mean_w + k_lee * (img[y, x] - mean_w)
            else:
                out[y, x] = mean_w
    
    return np.clip(out, 0, 255).astype(np.uint8)

def numpy_to_python(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_python(x) for x in obj)
    else:
        return obj

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)