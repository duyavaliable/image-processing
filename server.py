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
            
            # Get target class from form (luôn là 'all')
            target_class = request.form.get('target_class', 'all').strip().lower()
            print(f"[count_objects] target_class received: '{target_class}'")
            
            detected_objects = []  # Lưu danh sách đối tượng đã detect
            
            for i in (indices.flatten() if len(indices) else []):
                x, y, w, h = boxes[i]
                cls = class_ids[i] if i < len(class_ids) else -1
                label = CLASSES[cls] if (isinstance(CLASSES, list) and cls < len(CLASSES)) else f'class_{cls}'
                score = confidences[i]
                
                # Vẽ bounding box cho tất cả đối tượng
                cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_color, f"{label} {score:.2f}", (x, max(15, y-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Lưu thông tin đối tượng
                detected_objects.append({
                    'label': label,
                    'confidence': round(float(score), 2),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
            
            # THAY ĐỔI: không vẽ text overlay "Total: X" nữa
            # cv2.putText(img_color, text, (10, 30), ...) <-- XÓA dòng này
            
            # Trả về danh sách đối tượng thay vì count
            params = {
                'detected_objects': detected_objects,  # Danh sách đối tượng
                'total_detected': len(detected_objects),  # Số lượng (dùng cho params, không hiển thị)
                'backend': 'yolo_detect'
            }
            
            message = f'Detected {len(detected_objects)} objects' if len(detected_objects) > 0 else 'No objects detected'
            
            return jsonify({
                'dataURL': to_data_url_png(img_color), 
                'message': message, 
                'params': params
            }), 200

        # REMOVED: denoise auto
        # if mode == 'denoise': ...
        
        if mode == 'denoise_manual':
            # Manual denoise: user selects method
            img = ensure_gray_uint8_from_bytes(data)
            method_name = request.form.get('method', 'median')
            kernel = int(request.form.get('kernel', '5'))
            if kernel % 2 == 0:
                kernel += 1
            kernel = max(3, min(31, kernel))
            
            # HISTOGRAM ANALYSIS for noise detection
            H, W = img.shape
            total = float(H * W)
            n_black = int(np.sum(img == 0))
            n_white = int(np.sum(img == 255))
            extreme_ratio = float((n_black + n_white) / total)
            
            flat = img.flatten().astype(np.float32)
            mean_val = float(np.mean(flat))
            var = float(np.var(flat))
            cv_val = float(np.std(flat) / (mean_val + 1e-12))
            
            # Suggest method based on histogram features
            suggested_method = None
            if extreme_ratio > 0.02:
                suggested_method = 'median'  # salt-pepper
            elif cv_val > 0.28:
                suggested_method = 'lee'  # speckle
            elif 100 < var < 2000:
                suggested_method = 'bilateral'  # gaussian
            else:
                suggested_method = method_name  # use user's choice
            
            # Apply selected method
            if method_name == 'median':
                img_denoised = cv2.medianBlur(img, kernel)
                method = f'Median (k={kernel})'
            elif method_name == 'bilateral':
                sigma = max(10, min(100, kernel * 10))
                img_denoised = cv2.bilateralFilter(img, kernel, sigma, sigma)
                method = f'Bilateral (d={kernel}, sigma={sigma})'
            elif method_name == 'nlm':
                img_denoised = cv2.fastNlMeansDenoising(img, None, h=kernel*2, templateWindowSize=7, searchWindowSize=21)
                method = f'Non-local Means (h={kernel*2})'
            elif method_name == 'lee':
                img_denoised = lee_filter(img, window_size=kernel)
                method = f'Lee Filter (w={kernel})'
            else:
                img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
                method = 'Bilateral (default)'
            
            params = {
                'method': method,
                'kernel': kernel,
                'histogram': {
                    'extreme_ratio': round(extreme_ratio, 4),
                    'variance': round(var, 2),
                    'cv': round(cv_val, 4),
                    'suggested_method': suggested_method
                }
            }
            return jsonify({'dataURL': to_data_url_png(img_denoised), 'message': f'Manual denoise: {method}', 'params': params}), 200
        
        if mode == 'histogram_eq':
            # Histogram Equalization (File 03)
            img = ensure_gray_uint8_from_bytes(data)
            img_eq = cv2.equalizeHist(img)
            params = {'method': 'Histogram Equalization'}
            return jsonify({'dataURL': to_data_url_png(img_eq), 'message': 'Histogram Equalized', 'params': params}), 200
        
        if mode == 'sharpening':
            # Sharpening: preserve color if input is color
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                # Fallback: try grayscale
                img = ensure_gray_uint8_from_bytes(data)
                is_color = False
            else:
                is_color = True
            
            sharp_type = request.form.get('sharp_type', 'laplacian')
            
            # Apply sharpening (works for both grayscale and color)
            if sharp_type == 'laplacian':
                # Laplacian sharpening: f - ∇²f
                # KHÔNG sử dụng A parameter
                if is_color:
                    # Apply Laplacian per channel
                    img_sharp = np.zeros_like(img, dtype=np.uint8)
                    for c in range(3):
                        laplacian = cv2.Laplacian(img[:, :, c], cv2.CV_64F)
                        # f - laplacian (standard Laplacian sharpening)
                        img_sharp[:, :, c] = np.clip(img[:, :, c].astype(np.float64) - laplacian, 0, 255).astype(np.uint8)
                else:
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)
                    # f - laplacian (standard Laplacian sharpening)
                    img_sharp = np.clip(img.astype(np.float64) - laplacian, 0, 255).astype(np.uint8)
                method = 'Laplacian'
            else:
                img_sharp = img
                method = 'No sharpening'
            
            params = {'method': method, 'color': is_color}
            return jsonify({'dataURL': to_data_url_png(img_sharp), 'message': f'Sharpened: {method}', 'params': params}), 200
        
        if mode == 'frequency_filter':
            # Frequency domain filters (File 05: Lowpass/Highpass Ideal/Butterworth/Gaussian)
            img = ensure_gray_uint8_from_bytes(data)
            filter_type = request.form.get('filter_type', 'lowpass_gaussian')
            cutoff = float(request.form.get('cutoff', '30'))
            order_n = int(request.form.get('order', '2'))
            
            H, W = img.shape
            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Create frequency mask
            cy, cx = H // 2, W // 2
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            
            if 'lowpass' in filter_type:
                if 'ideal' in filter_type:
                    mask = (dist <= cutoff).astype(np.float32)
                elif 'butterworth' in filter_type:
                    mask = 1 / (1 + (dist / (cutoff + 1e-12))**(2*order_n))
                else:  # gaussian
                    mask = np.exp(-(dist**2) / (2 * cutoff**2))
                method = f'Lowpass {filter_type.split("_")[1].capitalize()} (D0={cutoff})'
            else:  # highpass
                if 'ideal' in filter_type:
                    mask = (dist > cutoff).astype(np.float32)
                elif 'butterworth' in filter_type:
                    mask = 1 / (1 + (cutoff / (dist + 1e-12))**(2*order_n))
                else:  # gaussian
                    mask = 1 - np.exp(-(dist**2) / (2 * cutoff**2))
                method = f'Highpass {filter_type.split("_")[1].capitalize()} (D0={cutoff})'
            
            # Apply mask
            mask = np.stack([mask, mask], axis=-1)
            dft_shift_filtered = dft_shift * mask
            dft_ishift = np.fft.ifftshift(dft_shift_filtered)
            img_back = cv2.idft(dft_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            
            # Normalize
            img_min, img_max = img_back.min(), img_back.max()
            if img_max > img_min:
                img_back = ((img_back - img_min) / (img_max - img_min) * 255.0)
            img_filtered = np.clip(img_back, 0, 255).astype(np.uint8)
            
            params = {'method': method, 'cutoff': cutoff, 'order': order_n}
            return jsonify({'dataURL': to_data_url_png(img_filtered), 'message': f'Frequency filtered: {method}', 'params': params}), 200
        
        if mode == 'wiener':
            # Wiener filter (File 07) — simplified (assume known noise variance)
            img = ensure_gray_uint8_from_bytes(data)
            noise_var = float(request.form.get('noise_var', '100'))
            
            # Estimate signal variance
            signal_var = float(np.var(img.astype(np.float32)))
            
            # Wiener filter in spatial domain (approximation)
            # true Wiener requires PSF — here use bilateral as approximation
            sigma_color = max(10, min(100, int(np.sqrt(noise_var))))
            sigma_space = sigma_color
            img_wiener = cv2.bilateralFilter(img, 9, sigma_color, sigma_space)
            
            params = {'method': 'Wiener (approx via Bilateral)', 'noise_var': noise_var, 'signal_var': signal_var}
            return jsonify({'dataURL': to_data_url_png(img_wiener), 'message': 'Wiener filtered', 'params': params}), 200

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
                
                if gamma_param is None:
                    return jsonify({'error': 'Missing required parameter: gamma'}), 400
                
                gamma_val = float(gamma_param)

                r = img.astype(np.float32) / 255.0
                s = np.power(r, gamma_val)
                img = np.clip((s * 255.0), 0, 255).astype(np.uint8)
                params = {'gamma': float(gamma_val)}

            elif mode == 'negative':
                img = 255 - img

            elif mode == 'threshold':
                T = int(request.form.get('threshold') or request.form.get('T') or 128)
                _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
                params['threshold'] = T  # add to response params

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
                # Require manual r1/s1/r2/s2
                has_manual = all(k in request.form for k in ['r1', 's1', 'r2', 's2'])
                if not has_manual:
                    return jsonify({'error': 'Missing required parameters: r1, s1, r2, s2'}), 400
                
                r1 = int(request.form.get('r1'))
                s1 = int(request.form.get('s1'))
                r2 = int(request.form.get('r2'))
                s2 = int(request.form.get('s2'))

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
                
                params = {'r1': r1, 's1': s1, 'r2': r2, 's2': s2}

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
    Primary: Frequency Domain Processing
    Fallback: Histogram features (extreme_ratio, variance, CV)
    """
    H, W = img.shape
    total = float(H * W)

    # 1. Salt-pepper: extreme pixels + spatial randomness check
    n_black = int(np.sum(img == 0))
    n_white = int(np.sum(img == 255))
    extreme_ratio = float((n_black + n_white) / total)
    
    # STRICTER threshold: only if extreme_ratio > 2% (natural images rarely have >2% pure black/white)
    if extreme_ratio > 0.02:
        # Additional check: spatial distribution (salt-pepper is random, not clustered)
        # Simple heuristic: if extreme pixels are clustered (e.g., all in one corner) -> NOT salt-pepper
        extreme_mask = (img == 0) | (img == 255)
        if extreme_mask.any():
            # Count connected components of extreme pixels
            num_labels, _ = cv2.connectedComponents(extreme_mask.astype(np.uint8), connectivity=8)
            # If few large clusters -> likely natural extreme values (shadows/highlights), NOT noise
            # Salt-pepper has many small isolated pixels -> num_labels should be high
            if num_labels > (n_black + n_white) * 0.3:  # at least 30% of extreme pixels are isolated
                return ('salt_pepper', {'extreme_ratio': extreme_ratio, 'black': n_black, 'white': n_white, 'num_clusters': int(num_labels)})
        else:
            # fallthrough if no extreme pixels (should not happen if extreme_ratio > 0.02)
            pass

    # 2. FREQUENCY DOMAIN (PRIMARY METHOD)
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)

    # Exclude DC component
    cy, cx = H // 2, W // 2
    r_dc = max(3, int(min(H, W) * 0.01))
    mask = np.ones_like(mag_log, dtype=bool)
    mask[max(0, cy-r_dc):min(H, cy+r_dc), max(0, cx-r_dc):min(W, cx+r_dc)] = False

    mag_vals = mag_log[mask]
    mean_mag = float(np.mean(mag_vals)) if mag_vals.size else 0.0
    std_mag = float(np.std(mag_vals)) if mag_vals.size else 0.0

    # Find peaks
    peak_thresh = mean_mag + 3.5 * std_mag
    peaks_idx = np.where((mag_log > peak_thresh) & mask)
    num_peaks = int(peaks_idx[0].size)

    # Compute peak energy ratio
    flat_mag = mag_vals.flatten()
    total_energy = float(np.sum(flat_mag**2) + 1e-12)
    peak_energy = 0.0
    if num_peaks > 0:
        peak_coords = list(zip(peaks_idx[0].tolist(), peaks_idx[1].tolist()))
        K = min(50, num_peaks)
        peak_vals = [(mag_log[y,x], y, x) for (y,x) in peak_coords]
        peak_vals.sort(reverse=True, key=lambda t: t[0])
        peak_energy = float(sum((v**2) for (v,_,_) in peak_vals[:K]))
    peak_energy_ratio = float(peak_energy / total_energy)

    # Decision: periodic if strong peaks with significant energy
    if num_peaks >= 6 and peak_energy_ratio > 0.015:
        top_coords = [(int(y), int(x)) for (v,y,x) in peak_vals[:8]]
        return ('periodic', {
            'num_peaks': num_peaks,
            'peak_energy_ratio': round(peak_energy_ratio, 6),
            'peak_coords': top_coords
        })

    # 3. Histogram fallback: Gaussian vs speckle
    flat = img.flatten().astype(np.float32)
    mean_val = float(np.mean(flat))
    var = float(np.var(flat))
    cv_val = float(np.std(flat) / (mean_val + 1e-12))

    if cv_val > 0.28:
        return ('speckle', {'cv': round(cv_val, 4), 'variance': round(var, 4)})
    
    if 100 < var < 2000:
        return ('gaussian', {'mean': round(mean_val, 4), 'variance': round(var, 4)})

    return ('unknown', {'variance': round(var, 4), 'cv': round(cv_val, 4)})

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