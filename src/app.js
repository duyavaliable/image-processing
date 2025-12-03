const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const preview = document.getElementById('preview');
const resultText = document.getElementById('resultText');



// prevent default behavior for drag/drop on document to avoid browser opening files
['dragenter','dragover','dragleave','drop'].forEach(ev=>{
    document.addEventListener(ev, e=>e.preventDefault());
});

// drop-area level handling (visual feedback + drop)
;['dragenter','dragover'].forEach(evt=>{
    dropArea.addEventListener(evt, e=>{
        e.preventDefault();
        dropArea.classList.add('dragover');
        dropArea.classList.add('highlight');
    });
});
;['dragleave','drop'].forEach(evt=>{
    dropArea.addEventListener(evt, e=>{
        e.preventDefault();
        dropArea.classList.remove('dragover');
        dropArea.classList.remove('highlight');
    });
});

dropArea.addEventListener('drop', e=>{
    const dt = e.dataTransfer;
    if(!dt) return;
    const file = dt.files[0];
    if(file) handleFile(file);
});

fileInput.addEventListener('change', ()=>{ if(fileInput.files[0]) handleFile(fileInput.files[0]); });

// optional controls from the markup
const clearBtn = document.getElementById('clearBtn');
const controls = document.getElementById('controls');
const modeSelect = document.getElementById('modeSelect');
const gammaInput = document.getElementById('gammaInput');
const gammaAuto = document.getElementById('gammaAuto');
const piecewiseControls = document.getElementById('piecewiseControls');
const r1Input = document.getElementById('r1Input');
const s1Input = document.getElementById('s1Input');
const r2Input = document.getElementById('r2Input');
const s2Input = document.getElementById('s2Input');
if (clearBtn) clearBtn.addEventListener('click', ()=> resetAll());

const uploadBtn = document.getElementById('uploadBtn');
if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', ()=> fileInput.click());
}

// cache for loaded libraries / model
let cached = {
    tf: null,
    mobilenet: null,
    model: null,
    loadingPromise: null
};

// --- ADD: keep original File for server uploads ---
let lastOriginalFile = null;
let lastOriginalDataURL = null;

// --- NEW: manual-override flags for gamma & piecewise ---
let forceManualGamma = false;
let forceManualPiecewise = false;

// collect extras to send to server (respects one-time manual overrides)
//xử lý auto tự động
function collectExtrasForMode(mode){
    const extras = {};
    if (mode === 'gamma') {
        // Auto has priority: if checkbox checked => send auto flag.
        if (gammaAuto && gammaAuto.checked) {
            extras.auto = '1';
        } else {
            // Auto is off => use manual gamma if present (or one-time override)
            const v = parseFloat((gammaInput && gammaInput.value) || '');
            if (!isNaN(v) && (forceManualGamma || v !== 0)) extras.gamma = v;
        }
    } else if (mode === 'piecewise') {
        // Auto has priority: if checked -> send auto flag.
        if (gammaAuto && gammaAuto.checked) {
            extras.auto = '1';
        } else {
            // Auto off -> use manual piecewise params (or one-time override)
            const r1 = parseInt((r1Input && r1Input.value) || 70, 10);
            const s1 = parseInt((s1Input && s1Input.value) || 10, 10);
            const r2 = parseInt((r2Input && r2Input.value) || 180, 10);
            const s2 = parseInt((s2Input && s2Input.value) || 245, 10);
            if (!isNaN(r1)) extras.r1 = r1;
            if (!isNaN(s1)) extras.s1 = s1;
            if (!isNaN(r2)) extras.r2 = r2;
            if (!isNaN(s2)) extras.s2 = s2;
        }
    }
    // console.log('[collectExtrasForMode] mode=', mode, 'extras=', extras);
    return extras;
}

// handle mode change
if (modeSelect) {
    modeSelect.addEventListener('change', async () => {
        const mode = modeSelect.value || 'gray8';
        // console.log('[modeSelect.change] mode=', mode, 'forceManualGamma=', forceManualGamma, 'forceManualPiecewise=', forceManualPiecewise);

        // show/hide piecewise controls only when mode is piecewise
        if (piecewiseControls) piecewiseControls.style.display = (mode === 'piecewise') ? '' : 'none';

        // Auto checkbox should be visible only for gamma or piecewise modes
        // gammaAuto is the checkbox element; its parent label contains text + input
        if (gammaAuto && gammaAuto.parentElement) {
            gammaAuto.parentElement.style.display = (mode === 'gamma' || mode === 'piecewise') ? '' : 'none';
        }

        // gammaInput visible only when mode is gamma AND Auto is unchecked
        if (gammaInput) {
            gammaInput.style.display = (mode === 'gamma' && !(gammaAuto && gammaAuto.checked)) ? '' : 'none';
        }
        
         if (!lastOriginalDataURL) return; // chưa có ảnh để xử lý

        const selectedMode = modeSelect.value || 'gray8';
        // if original file is TIFF (browser cannot render) -> server-process directly
        const isTiffChange = lastOriginalFile && (lastOriginalFile.type === 'image/tiff' || /\.tiff?$/i.test(lastOriginalFile.name));
        if (isTiffChange) {
            resultText.textContent = 'Đang xử lý ảnh (TIFF) trên server...';
            dropArea.classList.add('loading');
            try {
                // collect extras consistently (handles gamma auto/manual and piecewise)
                const extras = collectExtrasForMode(mode);
                const resp = await uploadFileToServer(lastOriginalFile, mode, extras);
                if (resp && resp.dataURL) {
                    showPreview(resp.dataURL);
                    resultText.textContent = resp.message || 'Đã xử lý trên server';
                } else {
                    resultText.textContent = 'Không có dữ liệu trả về từ server.';
                }
            } catch (err) {
                resultText.textContent = 'Lỗi khi áp dụng chế độ: ' + (err && err.message ? err.message : String(err));
            } finally {
                dropArea.classList.remove('loading');
            }
            return;
        }
        resultText.textContent = 'Áp dụng chế độ...';
        dropArea.classList.add('loading');

        try {
            // For negative: do NOT process on client, send original file to server.
            if (mode === 'negative') {
                // preview original immediately
                showPreview(lastOriginalDataURL);
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, mode);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        resultText.textContent = resp.message || 'Đã xử lý trên server';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi khi áp dụng chế độ: ' + (err && err.message ? err.message : String(err));
                }
                return;
            }
            // For gamma: send original file to server (server will auto-detect or use provided gamma)
            if (mode === 'gamma') {
                showPreview(lastOriginalDataURL);
                // use collectExtrasForMode to ensure auto/manual behavior is consistent
                const extras = collectExtrasForMode('gamma');
                // console.log('[handleFile] gamma branch extras=', extras);
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, 'gamma', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        // show message + params so you can see rank/downscale used
                        const msg = resp.message || 'Đã xử lý trên server';
                        const p = resp.params ? ('\n' + JSON.stringify(resp.params)) : '';
                        resultText.textContent = msg + p;
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi khi áp dụng chế độ: ' + (err && err.message ? err.message : String(err));
                } finally {
                    // clear one-time manual flags after upload
                    forceManualGamma = false;
                    forceManualPiecewise = false;
                    dropArea.classList.remove('loading');
                }
                return;
            }
            // For log: do NOT process on client, send original file to server.
            if (mode === 'log' || mode === 'logarithmic') {
                showPreview(lastOriginalDataURL);
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, 'log');
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        resultText.textContent = resp.message || 'Đã xử lý trên server';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi khi áp dụng chế độ: ' + (err && err.message ? err.message : String(err));
                }
                return;
            }

            // For detect: upload original file and show detection result returned by server
            if (mode === 'detect') {
                showPreview(lastOriginalDataURL);
                try {
                    // extras can include detection threshold later if needed
                    const resp = await uploadFileToServer(lastOriginalFile, 'detect');
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || 'Đã phát hiện đối tượng') + paramsText;
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho detect.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi khi nhận diện: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // existing client-side branches for other modes
            let processed = lastOriginalDataURL;
            if (mode === 'gray8') {
                processed = await toGray8DataURL(lastOriginalDataURL);
            } else if (mode === 'threshold') {
                processed = await toThresholdDataURL(lastOriginalDataURL);
            } else if (mode === 'log' || mode === 'logarithmic') {
                processed = await toLogDataURL(lastOriginalDataURL, 2.0);
            }
            showPreview(processed);
            try {
                const resp = await uploadFileToServer(dataURLToBlob(processed), mode);
                if (resp && resp.dataURL) {
                    showPreview(resp.dataURL); 
                    resultText.textContent = resp.message || 'Đã xử lý trên server';
                }
            } catch (err) {
                resultText.textContent = 'Lỗi khi áp dụng chế độ: ' + (err && err.message ? err.message : String(err));
            }
        } finally {
            // clear one-time manual flags after any client-side branch that reaches here
            forceManualGamma = false;
            forceManualPiecewise = false;
            dropArea.classList.remove('loading');
        }
    });
}

// Enter in gamma -> one-time manual override and re-run processing immediately
if (gammaInput) {
    gammaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.keyCode === 13) {
            e.preventDefault();
            // console.log('[gammaInput] Enter pressed, value=', gammaInput.value);
            forceManualGamma = true;
            if (gammaAuto && gammaAuto.checked) {
                gammaAuto.checked = false;
                // update visibility immediately
                if (gammaInput) gammaInput.style.display = (modeSelect && modeSelect.value === 'gamma' && !gammaAuto.checked) ? '' : 'none';
            }
            // ensure mode is gamma
            if (modeSelect) modeSelect.value = 'gamma';
            // re-run processing by calling handleFile with lastOriginalFile (if available)
            if (lastOriginalFile) {
                // console.log('[gammaInput] re-running handleFile, file=', lastOriginalFile.name);
                handleFile(lastOriginalFile).catch(()=>{ /* ignore */ });
            } else if (modeSelect) {
                // console.log('[gammaInput] dispatching change (no lastOriginalFile)');
                modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
}

// Enter in piecewise inputs -> one-time manual override and re-run processing immediately
function piecewiseEnterHandler(e){
    if (e.key === 'Enter' || e.keyCode === 13) {
        e.preventDefault();
        forceManualPiecewise = true;
        // user intends manual -> turn Auto OFF so manual params are used
        if (gammaAuto && gammaAuto.checked) {
            gammaAuto.checked = false;
            if (piecewiseControls) piecewiseControls.style.display = (modeSelect && modeSelect.value === 'piecewise' && !gammaAuto.checked) ? '' : 'none';
            if (gammaInput) gammaInput.style.display = (modeSelect && modeSelect.value === 'gamma' && !gammaAuto.checked) ? '' : 'none';
        }
        if (modeSelect) modeSelect.value = 'piecewise';
        if (lastOriginalFile) {
            handleFile(lastOriginalFile).catch(()=>{ /* ignore */ });
        } else if (modeSelect) {
            modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
}
if (r1Input) r1Input.addEventListener('keydown', piecewiseEnterHandler);
if (s1Input) s1Input.addEventListener('keydown', piecewiseEnterHandler);
if (r2Input) r2Input.addEventListener('keydown', piecewiseEnterHandler);
if (s2Input) s2Input.addEventListener('keydown', piecewiseEnterHandler);

// Auto checkbox: when toggled, hide gamma input and (optionally) hide piecewise inputs.
// Show Auto only for gamma/piecewise is handled in modeSelect.change above.
if (gammaAuto) {
    // initial visibility for gammaInput and piecewiseControls
    if (gammaInput) gammaInput.style.display = (modeSelect && modeSelect.value === 'gamma' && !gammaAuto.checked) ? '' : 'none';
    if (piecewiseControls) piecewiseControls.style.display = (modeSelect && modeSelect.value === 'piecewise' && !gammaAuto.checked) ? '' : 'none';

    gammaAuto.addEventListener('change', () => {
        // gamma input visible only when mode is gamma and Auto unchecked
        if (gammaInput) gammaInput.style.display = (modeSelect && modeSelect.value === 'gamma' && !gammaAuto.checked) ? '' : 'none';
        // piecewise controls visible only when mode is piecewise and Auto unchecked
        if (piecewiseControls) piecewiseControls.style.display = (modeSelect && modeSelect.value === 'piecewise' && !gammaAuto.checked) ? '' : 'none';

        // re-run processing for active mode so server uses auto/manual as needed
        if (modeSelect && (modeSelect.value === 'gamma' || modeSelect.value === 'piecewise')) {
            if (lastOriginalFile) handleFile(lastOriginalFile).catch(()=>{});
            else modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });
}

// Global Enter: if mode is gamma and gammaInput has a numeric value -> force manual override once
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.keyCode === 13) {
        if (modeSelect && modeSelect.value === 'gamma' && gammaInput) {
            const v = parseFloat((gammaInput.value || '').trim());
            if (!isNaN(v) && v !== 0) {
                forceManualGamma = true;
                modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }
});

//hien thi preview anh 
export async function handleFile(file){
    if(!file) return;
    if(!file.type || !file.type.startsWith('image/')) {
        resultText.textContent = 'Vui lòng chọn file ảnh.';
        return;
    }
    const MAX_MB = 8;
    if(file.size > MAX_MB * 1024 * 1024){
        resultText.textContent = `Ảnh quá lớn (>${MAX_MB}MB). Vui lòng chọn ảnh nhỏ hơn.`;
        return;
    }

    const reader = new FileReader();
    reader.onload = async e=>{
        const dataURL = e.target.result;
        lastOriginalDataURL = dataURL;
        lastOriginalFile = file;

        // attempt lightweight local preview (ignore errors)
        try { showInputPreview(file); } catch(e){ /* ignore */ }

        const isTiff = file && (file.type === 'image/tiff' || /\.tiff?$/i.test(file.name));
        const mode = (modeSelect && modeSelect.value) ? modeSelect.value : 'gray8';

        // determine processed preview to show (client-side where implemented)
        let processedDataURL = dataURL;
        if (isTiff) {
            // for TIFF rely on server preview; keep dataURL as-is for initial preview
            processedDataURL = dataURL;
        } else {
            if (mode === 'gray8') {
                processedDataURL = await toGray8DataURL(dataURL);
            } else if (mode === 'threshold') {
                processedDataURL = await toThresholdDataURL(dataURL);
            } else if (mode === 'log' || mode === 'logarithmic') {
                // best-effort preview client-side; fall back to original if conversion fails
                try { processedDataURL = await toLogDataURL(dataURL, 2.0); } catch(e){ processedDataURL = dataURL; }
            } else if (mode === 'median') {
                // allowed client-side median preview (optional)
                processedDataURL = await toMedianDataURL(dataURL, 3);
            } else {
                // for negative/gamma/piecewise and other server-side modes use original preview
                processedDataURL = dataURL;
            }
        }

        showPreview(processedDataURL);

        // single high-level upload + response handling
        resultText.textContent = 'Đang tải lên server...';
        dropArea.classList.add('loading');
        try {
            // include median so server receives original file and applies OpenCV medianBlur
            const useOriginalFile = (mode === 'negative' || mode === 'log' || mode === 'logarithmic' || mode === 'piecewise' || mode === 'gamma' || mode === 'median' || isTiff);
            const uploadTarget = useOriginalFile && lastOriginalFile ? lastOriginalFile : dataURLToBlob(processedDataURL);
            const extras = collectExtrasForMode(mode);
            if (mode === 'median') {
                // stronger defaults for heavy salt&pepper; tune as needed
                extras.k = extras.k || 5;
                extras.passes = extras.passes || 2;
                extras.morph = extras.morph || 3;
                // extras.gauss = 3; // enable if you want pre-smoothing
            }
            const resp = await uploadFileToServer(uploadTarget, mode, extras);
            // debug: print full server JSON response so you can inspect params/message
            console.debug('[handleFile] server response:', resp);
            if (resp && resp.dataURL) {
                showPreview(resp.dataURL);
                // show message + params so you can see rank/downscale used
                const msg = resp.message || 'Đã xử lý trên server';
                const p = resp.params ? ('\n' + JSON.stringify(resp.params)) : '';
                resultText.textContent = msg + p;
            } else {
                resultText.textContent = 'Ảnh đã tải lên nhưng server không trả dataURL.';
            }
        } catch (err) {
            resultText.textContent = 'Lỗi server: ' + (err && err.message ? err.message : String(err));
        } finally {
            dropArea.classList.remove('loading');
        }
    };
    reader.onerror = ()=>{
        resultText.textContent = 'Không thể đọc file.';
    };
    reader.readAsDataURL(file);
}

export function resetAll(){
    try{
        preview.innerHTML = '';
        resultText.textContent = 'Chưa có ảnh';
        if(fileInput) fileInput.value = '';
    }catch(e){
        // ignore
    }
}





// helper: create image element and wait load
// tai anh từ dataURL, URL http(s), hoặc Blob/File
function loadImageElement(src){
    // Accept dataURL string, http(s) URL, or Blob/File.
    return new Promise((resolve, reject)=>{
        const img = new Image();
        let objectUrl = null;
        try {
            // If src is a Blob/File, create an object URL
            if (src instanceof Blob) {
                objectUrl = URL.createObjectURL(src);
                img.src = objectUrl;
            } else if (typeof src === 'string') {
                // set crossOrigin only for remote HTTP(S) resources
                if (src.startsWith('http://') || src.startsWith('https://')) {
                    img.crossOrigin = 'anonymous';
                }
                img.src = src;
            } else {
                // unsupported type
                return reject(new Error('Invalid image source type'));
            }
        } catch (err) {
            if (objectUrl) { URL.revokeObjectURL(objectUrl); }
            return reject(new Error('Không thể tải ảnh để phân tích.'));
        }
        img.onload = ()=>{
            if (objectUrl) { URL.revokeObjectURL(objectUrl); }
            resolve(img);
        };
        img.onerror = (ev)=>{
            if (objectUrl) { URL.revokeObjectURL(objectUrl); }
            // debug help: log actual src type/length to console (can be removed later)
            try { console.debug('loadImageElement failed for src type:', typeof src, src && (src.size || src.length || (''+src).slice(0,64))); } catch(e){}
            reject(new Error('Không thể tải ảnh để phân tích.'));
        };
    });
}

//grayscale 8-bit
async function toGray8DataURL(srcDataURL){
    const img = await loadImageElement(srcDataURL);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    for(let i = 0; i < data.length; i += 4){
        const r = data[i], g = data[i+1], b = data[i+2];
        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        data[i] = gray;
        data[i+1] = gray;
        data[i+2] = gray;
        // keep alpha channel as is
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

// negative image 
async function toNegativeDataURL(srcDataURL){
    // convert to grayscale then invert (match Python: neg = 255 - img)
    const img = await loadImageElement(srcDataURL);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    for(let i = 0; i < data.length; i += 4){
        const r = data[i], g = data[i+1], b = data[i+2];
        // ensure grayscale luminance first
        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        const inv = 255 - gray;
        data[i] = inv;
        data[i+1] = inv;
        data[i+2] = inv;
       
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

//Thresholding (Binary)
async function toThresholdDataURL(srcDataURL, T = 128){
    const img = await loadImageElement(srcDataURL);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    for(let i = 0; i < data.length; i += 4){
        const r = data[i], g = data[i+1], b = data[i+2];
        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        const v = gray > T ? 255 : 0;
        data[i] = v;
        data[i+1] = v;
        data[i+2] = v;
        // alpha unchanged
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

//Logarithmic transformation
async function toLogDataURL(srcDataURL, c = 2.0){
    const img = await loadImageElement(srcDataURL);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    // First pass: compute s = log1p(c * r) per pixel, store in float array and track max
    const sArr = new Float32Array((w * h));
    let maxS = 0.0;
    let idx = 0;
    for(let i = 0; i < data.length; i += 4){
        const r = data[i], g = data[i+1], b = data[i+2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        const norm = gray / 255.0;
        const s = Math.log1p(c * norm); // natural log
        sArr[idx++] = s;
        if(s > maxS) maxS = s;
    }
    // normalize and write back to imageData
    const denom = (maxS + 1e-12);
    idx = 0;
    for(let i = 0; i < data.length; i += 4){
        const v = Math.round((sArr[idx++] / denom) * 255);
        data[i] = data[i+1] = data[i+2] = v;
        // keep alpha unchanged
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

// Median filter (salt-and-pepper denoise) - client-side only
// Xử lý ảnh khi bị nhiễu muối tiêu
async function toMedianDataURL(srcDataURL, kernelSize = 3) {
    const img = await loadImageElement(srcDataURL);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    const src = imageData.data;
    const out = new Uint8ClampedArray(src.length);

    // ensure odd kernel
    if (kernelSize % 2 === 0) kernelSize = kernelSize + 1;
    const k = Math.floor(kernelSize / 2);

    // For efficiency operate on grayscale luminance then write back as gray image
    const getIndex = (x, y) => ((y * w) + x) * 4;
    for (let yy = 0; yy < h; yy++) {
        for (let xx = 0; xx < w; xx++) {
            const vals = [];
            for (let dy = -k; dy <= k; dy++) {
                const ny = Math.min(h - 1, Math.max(0, yy + dy));
                for (let dx = -k; dx <= k; dx++) {
                    const nx = Math.min(w - 1, Math.max(0, xx + dx));
                    const idx = getIndex(nx, ny);
                    // compute grayscale luminance (same formula used elsewhere)
                    const r = src[idx], g = src[idx + 1], b = src[idx + 2];
                    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                    vals.push(gray);
                }
            }
            vals.sort((a,b)=>a-b);
            const med = vals[Math.floor(vals.length/2)];
            const outIdx = getIndex(xx, yy);
            out[outIdx] = med;
            out[outIdx+1] = med;
            out[outIdx+2] = med;
            out[outIdx+3] = src[outIdx+3]; // preserve alpha
        }
    }

    // put back and return PNG dataURL
    imageData.data.set(out);
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

function formatPredictions(preds){
    if(!Array.isArray(preds) || preds.length === 0) return 'Không có kết quả.';
    return preds.map((p,i)=>`${i+1}. ${p.className} — ${(p.probability*100).toFixed(2)}%`).join('\n');
}

// --- ADD: convert dataURL to Blob ---
function dataURLToBlob(dataURL){
    if(!dataURL || typeof dataURL !== 'string') throw new Error('Invalid dataURL');
    const parts = dataURL.split(',');
    const meta = parts[0] || '';
    const b64 = parts[1] || '';
    const mimeMatch = meta.match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
    const binary = atob(b64);
    let n = binary.length;
    const u8 = new Uint8Array(n);
    while(n--) u8[n] = binary.charCodeAt(n);
    return new Blob([u8], { type: mime });
}

// --- ADD: upload File or Blob to server API /api/process ---
async function uploadFileToServer(fileOrBlob, mode='gray8', extras = {}){
    // console.log('[uploadFileToServer] start', { mode, extras, payloadType: (fileOrBlob && fileOrBlob.constructor && fileOrBlob.constructor.name) || typeof fileOrBlob });
    const form = new FormData();
    let payload;
    if (fileOrBlob instanceof Blob && !(fileOrBlob instanceof File)) {
        // give a filename so server's FieldStorage / Flask sees a name
        payload = new File([fileOrBlob], 'upload.png', { type: fileOrBlob.type || 'image/png' });
    } else {
        payload = fileOrBlob;
    }
    form.append('file', payload);
    form.append('mode', mode || 'gray8');
    // extras: e.g. { threshold: 128, c: 2.0 }
    for (const k in extras) {
        if (Object.prototype.hasOwnProperty.call(extras, k) && extras[k] != null) {
            form.append(k, String(extras[k]));
        }
    }

    let res;
    try {
        res = await fetch('/api/process', { method: 'POST', body: form });
    } catch (networkErr) {
        throw new Error('Network error: ' + (networkErr && networkErr.message ? networkErr.message : String(networkErr)));
    }
    if (!res.ok) {
        // try parse JSON error first
        let errText = res.statusText;
        try {
            const j = await res.json().catch(()=>null);
            if (j && j.error) errText = j.error;
        } catch(e){
            // fallback to text
            const t = await res.text().catch(()=>null);
            if (t) errText = t;
        }
        // console.error('[uploadFileToServer] server returned error', { status: res.status, errText });
        throw new Error(errText || `HTTP ${res.status}`);
    }
    const j = await res.json();
    // console.log('[uploadFileToServer] server json keys=', j && Object.keys(j), 'message=', j && j.message);
    return j;
}

// show original image in the input / drop area for quick visual feedback . Hiện thị ảnh ở Input 
function showInputPreview(src){
    if(!dropArea) return;
    const existing = dropArea.querySelector('.input-preview');
    if(existing) existing.remove();

    const img = document.createElement('img');
    img.className = 'input-preview';
    img.alt = 'input preview';
    img.style.maxWidth = '160px';
    img.style.maxHeight = '120px';
    img.style.objectFit = 'cover';
    img.style.borderRadius = '6px';
    img.style.margin = '6px';
    img.style.visibility = 'hidden';

    const controlsEl = dropArea.querySelector('#controls') || dropArea.firstChild;
    dropArea.insertBefore(img, controlsEl);

    // Helper: try to set src and await decode
    const trySet = async (url)=>{
        try {
            img.src = url;
            if (img.decode) await img.decode();
            img.style.visibility = 'visible';
            return true;
        } catch(e){
            return false;
        }
    };

    (async ()=>{
        // 1) If File/Blob and it's TIFF -> ask server to convert & return PNG dataURL
        if (src instanceof File || src instanceof Blob) {
            const t = (src.type || '').toLowerCase();
            const name = (src.name || '').toLowerCase();
            const isTiff = t === 'image/tiff' || /\.tiff?$/i.test(name);
            if (isTiff) {
                // show temporary placeholder (optional)
                resultText.textContent = 'Đang tạo preview từ server...';
                try {
                    const resp = await uploadFileToServer(src, 'gray8'); // server sẽ trả dataURL PNG
                    if (resp && resp.dataURL) {
                        await trySet(resp.dataURL);
                        resultText.textContent = 'Preview (từ server) sẵn sàng.';
                        return;
                    } else {
                        resultText.textContent = 'Không nhận được preview từ server.';
                    }
                } catch (err) {
                    console.error('[showInputPreview] server preview error', err);
                    resultText.textContent = 'Lỗi tạo preview từ server.';
                }
                img.remove();
                return;
            }
            // not TIFF: try objectURL
            try {
                const obj = URL.createObjectURL(src);
                const ok = await trySet(obj);
                if (ok) { URL.revokeObjectURL(obj); return; }
                URL.revokeObjectURL(obj);
            } catch(e){
                /* ignore */
            }
        }

        // 2) If string dataURL (or other string) -> try directly
        if (typeof src === 'string') {
            const ok = await trySet(src);
            if (ok) return;

            // if it was a data:image/tiff and failed, upload original file if available
            if (lastOriginalFile) {
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, 'gray8');
                    if (resp && resp.dataURL) {
                        await trySet(resp.dataURL);
                        resultText.textContent = 'Preview (từ server) sẵn sàng.';
                        return;
                    }
                } catch(e){}
            }
        }

        // fallback: remove preview img
        img.remove();
    })();
}

// show result preview with onload/onerror logging
export function showPreview(dataURL){
    if(!preview) return;
    preview.innerHTML = '';
    const img = document.createElement('img');
    img.alt = 'preview';
    img.style.maxWidth = '420px';
    img.style.maxHeight = '420px';
    img.style.borderRadius = '6px';
    img.onload = ()=> {};
    img.onerror = (e)=> {
        console.error('[showPreview] failed to display image', e, 'src:', (dataURL && (dataURL.slice ? dataURL.slice(0,80) : String(dataURL))));
    };
    try {
        img.src = dataURL;
    } catch(e){
        console.error('[showPreview] setting src failed', e);
    }
    preview.appendChild(img);
}

// ensure old export kept for code that used it
export { showPreview as showPreviewOld };