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
const gammaControls = document.getElementById('gammaControls');
const piecewiseControls = document.getElementById('piecewiseControls');
const r1Input = document.getElementById('r1Input');
const s1Input = document.getElementById('s1Input');
const r2Input = document.getElementById('r2Input');
const s2Input = document.getElementById('s2Input');
const countControls = document.getElementById('countControls');
const denoiseManualControls = document.getElementById('denoiseManualControls');
const denoiseMethodSelect = document.getElementById('denoiseMethodSelect');
const denoiseKernelInput = document.getElementById('denoiseKernelInput');
const sharpeningControls = document.getElementById('sharpeningControls');
const sharpeningTypeSelect = document.getElementById('sharpeningTypeSelect');
const freqFilterControls = document.getElementById('freqFilterControls');
const freqFilterTypeSelect = document.getElementById('freqFilterTypeSelect');
const freqCutoffInput = document.getElementById('freqCutoffInput');
const freqOrderInput = document.getElementById('freqOrderInput');
const thresholdControls = document.getElementById('thresholdControls');
const thresholdInput = document.getElementById('thresholdInput');
if (clearBtn) clearBtn.addEventListener('click', ()=> resetAll());

// Helper: update controls visibility based on mode
function updateControlsVisibility(mode) {
    // show/hide piecewise controls only when mode is piecewise
    if (piecewiseControls) piecewiseControls.style.display = (mode === 'piecewise') ? '' : 'none';

    // show/hide count controls only when mode is count_objects
    if (countControls) countControls.style.display = (mode === 'count_objects') ? 'inline-flex' : 'none';
    
    // show/hide denoise manual controls
    if (denoiseManualControls) denoiseManualControls.style.display = (mode === 'denoise_manual') ? 'inline-flex' : 'none';
    
    // show/hide sharpening controls
    if (sharpeningControls) sharpeningControls.style.display = (mode === 'sharpening') ? 'inline-flex' : 'none';
    
    // show/hide frequency filter controls
    if (freqFilterControls) freqFilterControls.style.display = (mode === 'frequency_filter') ? 'inline-flex' : 'none';
    
    // show/hide threshold control
    if (thresholdControls) thresholdControls.style.display = (mode === 'threshold') ? 'inline-flex' : 'none';

    if (gammaControls) gammaControls.style.display = (mode === 'gamma') ? 'inline-flex' : 'none';
}

// Initialize on page load
function initializeControlsVisibility() {
    const mode = (modeSelect && modeSelect.value) || 'gray8';
    updateControlsVisibility(mode);
}
initializeControlsVisibility();

// Upload button: trigger file input click
const uploadBtn = document.getElementById('uploadBtn');
if (uploadBtn) {
    uploadBtn.addEventListener('click', () => {
        if (fileInput) {
            fileInput.click(); // programmatically open file picker
        }
    });
}

// handle mode change
if (modeSelect) {
    modeSelect.addEventListener('change', async () => {
        const mode = modeSelect.value || 'gray8';
        
        // Update controls visibility
        updateControlsVisibility(mode);
        
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
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang xử lý negative trên server...';
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
            
            // For threshold: send to server (avoid client-side JPEG decode issues)
            if (mode === 'threshold') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang xử lý threshold trên server...';
                try {
                    const T = parseInt((thresholdInput && thresholdInput.value) || '128', 10);
                    const extras = { threshold: T };
                    const resp = await uploadFileToServer(lastOriginalFile, 'threshold', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || `Thresholded (T=${T})`) + paramsText;
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi threshold: ' + (err && err.message ? err.message : String(err));
                }
                return;
            }

            // For gamma: send original file to server (server will auto-detect or use provided gamma)
            if (mode === 'gamma') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang xử lý gamma trên server...';
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
                    dropArea.classList.remove('loading');
                }
                return;
            }
            // For log: do NOT process on client, send original file to server.
            if (mode === 'log' || mode === 'logarithmic') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang xử lý log trên server...';
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
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang phát hiện đối tượng...';
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

            // For count_objects: upload original file + target class from input
            if (mode === 'count_objects') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang nhận diện đối tượng...';
                try {
                    const targetClass = 'all';
                    console.log('[count_objects] sending target_class:', targetClass);
                    const extras = { target_class: targetClass };
                    const resp = await uploadFileToServer(lastOriginalFile, 'count_objects', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        // THAY ĐỔI: chỉ hiển thị message, không hiển thị params
                        resultText.textContent = resp.message || 'Đã nhận diện đối tượng';
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho count_objects.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi khi nhận diện: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // For denoise_manual: upload original file + method/kernel from controls
            if (mode === 'denoise_manual') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang xử lý nhiễu...';
                try {
                    const method = (denoiseMethodSelect && denoiseMethodSelect.value) || 'median';
                    const kernel = parseInt((denoiseKernelInput && denoiseKernelInput.value) || '5', 10);
                    console.log('[denoise_manual] method:', method, 'kernel:', kernel);
                    const extras = { method: method, kernel: kernel };
                    const resp = await uploadFileToServer(lastOriginalFile, 'denoise_manual', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || 'Đã xử lý nhiễu') + paramsText;
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho denoise_manual.';
                    }
                } catch (err) {
                    console.error('[denoise_manual] upload/process error:', err);
                    resultText.textContent = 'Lỗi khi xử lý nhiễu: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // For histogram_eq: simple histogram equalization
            if (mode === 'histogram_eq') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang histogram equalization...';
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, 'histogram_eq');
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        resultText.textContent = resp.message || 'Histogram Equalized';
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho histogram_eq.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi histogram_eq: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // For sharpening: Laplacian / High-boost / Unsharp
            if (mode === 'sharpening') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang sharpening...';
                try {
                    const sharpType = (sharpeningTypeSelect && sharpeningTypeSelect.value) || 'laplacian';
                    console.log('[sharpening] type:', sharpType);
                    const extras = { sharp_type: sharpType };
                    const resp = await uploadFileToServer(lastOriginalFile, 'sharpening', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || 'Đã sharpening') + paramsText;
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho sharpening.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi sharpening: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // For frequency_filter: LPF/HPF (Ideal/Butterworth/Gaussian)
            if (mode === 'frequency_filter') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang lọc tần số...';
                try {
                    const filterType = (freqFilterTypeSelect && freqFilterTypeSelect.value) || 'lowpass_gaussian';
                    const cutoff = parseFloat((freqCutoffInput && freqCutoffInput.value) || '30');
                    const order = parseInt((freqOrderInput && freqOrderInput.value) || '2', 10);
                    console.log('[frequency_filter] type:', filterType, 'cutoff:', cutoff, 'order:', order);
                    const extras = { filter_type: filterType, cutoff: cutoff, order: order };
                    const resp = await uploadFileToServer(lastOriginalFile, 'frequency_filter', extras);
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || 'Đã lọc tần số') + paramsText;
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho frequency_filter.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi frequency_filter: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // For wiener: Wiener filter (deblur)
            if (mode === 'wiener') {
                preview.innerHTML = '<div class="loader"></div>';
                resultText.textContent = 'Đang Wiener filter...';
                try {
                    const resp = await uploadFileToServer(lastOriginalFile, 'wiener');
                    if (resp && resp.dataURL) {
                        showPreview(resp.dataURL);
                        const paramsText = resp.params ? ('\n' + JSON.stringify(resp.params, null, 2)) : '';
                        resultText.textContent = (resp.message || 'Đã Wiener filter') + paramsText;
                    } else {
                        resultText.textContent = 'Không có phản hồi từ server cho wiener.';
                    }
                } catch (err) {
                    resultText.textContent = 'Lỗi wiener: ' + (err && err.message ? err.message : String(err));
                } finally {
                    dropArea.classList.remove('loading');
                }
                return;
            }

            // existing client-side branches for other modes
            let processed = lastOriginalDataURL;
            if (mode === 'gray8') {
                processed = await toGray8DataURL(lastOriginalDataURL);
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



// Enter in denoiseKernelInput or denoiseMethodSelect -> re-run denoise_manual
if (denoiseKernelInput) {
    denoiseKernelInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.keyCode === 13) {
            e.preventDefault();
            console.log('[denoiseKernelInput Enter] kernel:', denoiseKernelInput.value);
            if (modeSelect) modeSelect.value = 'denoise_manual';
            if (lastOriginalFile) {
                handleFile(lastOriginalFile).catch(()=>{});
            } else if (modeSelect) {
                modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
    
    // Input event: re-process when user changes kernel (without pressing Enter)
    denoiseKernelInput.addEventListener('input', () => {
        console.log('[denoiseKernelInput input] kernel:', denoiseKernelInput.value);
        // Debounce: only re-process after user stops typing for 800ms
        clearTimeout(window.denoiseKernelDebounce);
        window.denoiseKernelDebounce = setTimeout(() => {
            if (modeSelect && modeSelect.value === 'denoise_manual' && lastOriginalFile) {
                console.log('[denoiseKernelInput input] debounced re-process, kernel:', denoiseKernelInput.value);
                handleFile(lastOriginalFile).catch(()=>{});
            }
        }, 800);
    });
}

// Change event for denoiseMethodSelect (user changes method dropdown)
if (denoiseMethodSelect) {
    denoiseMethodSelect.addEventListener('input', () => {
        console.log('[denoiseMethodSelect input] method:', denoiseMethodSelect.value);
        if (modeSelect && modeSelect.value === 'denoise_manual') {
            if (lastOriginalFile) {
                handleFile(lastOriginalFile).catch(()=>{});
            }
        }
    });
}

// Change event for sharpeningTypeSelect
if (sharpeningTypeSelect) {
    sharpeningTypeSelect.addEventListener('change', () => {
        console.log('[sharpeningTypeSelect change] type:', sharpeningTypeSelect.value);
        
        if (modeSelect && modeSelect.value === 'sharpening') {
            if (lastOriginalFile) {
                handleFile(lastOriginalFile).catch(()=>{});
            }
        }
    });
}

// Enter in freqCutoffInput or freqOrderInput -> re-run frequency_filter
function freqFilterEnterHandler(e) {
    if (e.key === 'Enter' || e.keyCode === 13) {
        e.preventDefault();
        console.log('[freqFilter Enter] cutoff:', freqCutoffInput.value, 'order:', freqOrderInput.value);
        if (modeSelect) modeSelect.value = 'frequency_filter';
        if (lastOriginalFile) {
            handleFile(lastOriginalFile).catch(()=>{});
        } else if (modeSelect) {
            modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
}
if (freqCutoffInput) freqCutoffInput.addEventListener('keydown', freqFilterEnterHandler);
if (freqOrderInput) freqOrderInput.addEventListener('keydown', freqFilterEnterHandler);

// Change event for freqFilterTypeSelect
if (freqFilterTypeSelect) {
    freqFilterTypeSelect.addEventListener('change', () => {
        console.log('[freqFilterTypeSelect change] type:', freqFilterTypeSelect.value);
        if (modeSelect && modeSelect.value === 'frequency_filter') {
            if (lastOriginalFile) {
                handleFile(lastOriginalFile).catch(()=>{});
            }
        }
    });
}

// Enter in thresholdInput -> re-run threshold
if (thresholdInput) {
    thresholdInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.keyCode === 13) {
            e.preventDefault();
            console.log('[thresholdInput Enter] T:', thresholdInput.value);
            if (modeSelect) modeSelect.value = 'threshold';
            if (lastOriginalFile) {
                handleFile(lastOriginalFile).catch(()=>{});
            } else if (modeSelect) {
                modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
}

// Global state: store last uploaded image data
let lastOriginalDataURL = false;
let lastOriginalFile = false;

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

        // EARLY CHECK: if JPEG is too large or has complex metadata, skip client processing
        const isJpeg = file && (file.type === 'image/jpeg' || /\.jpe?g$/i.test(file.name));
        const skipClientProcessing = isTiff || (isJpeg && file.size > 200 * 1024); // > 200KB JPEG

        // determine processed preview to show (client-side where implemented)
        let processedDataURL = dataURL;
        if (skipClientProcessing) {
            // for TIFF rely on server preview; keep dataURL as-is for initial preview
            processedDataURL = dataURL;
        } else {
            // IMPROVED: wrap all client-side processing in try-catch with fallback
            try {
                if (mode === 'gray8') {
                    processedDataURL = await toGray8DataURL(dataURL);
                } else if (mode === 'log' || mode === 'logarithmic') {
                    processedDataURL = await toLogDataURL(dataURL, 2.0);
                } else if (mode === 'median') {
                    processedDataURL = await toMedianDataURL(dataURL, 3);
                } else {
                    processedDataURL = dataURL;
                }
            } catch (clientProcessErr) {
                console.warn('[handleFile] client-side processing failed, using original dataURL:', clientProcessErr);
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
                resultText.textContent = 'Không có dữ liệu trả về từ server.';
            }
        } catch (err) {
            resultText.textContent = 'Lỗi khi tải lên server: ' + (err && err.message ? err.message : String(err));
        } finally {
            dropArea.classList.remove('loading');
        }
    };
    
    reader.onerror = (err) => {
        console.error('[handleFile] FileReader error:', err);
        resultText.textContent = 'Lỗi khi đọc file: ' + (err && err.message ? err.message : String(err));
        dropArea.classList.remove('loading');
    };
    
    reader.readAsDataURL(file);
}

export function resetAll(){
    try{
        preview.innerHTML = '';
        // cleanup input preview objectURL if exists
        const inputPreview = dropArea && dropArea.querySelector('.input-preview');
        if (inputPreview) {
            const objUrl = inputPreview.dataset.objectUrl;
            if (objUrl) {
                URL.revokeObjectURL(objUrl);
                delete inputPreview.dataset.objectUrl;
            }
            inputPreview.remove();
        }
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
            // SILENT: don't log here — caller will log if final fallback fails
            reject(new Error('Image decode failed (possibly corrupted JPEG/metadata)'));
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

// show original image in the input / drop area for quick visual feedback
function showInputPreview(src){
    console.log('[showInputPreview] START', {
        srcType: typeof src,
        srcConstructor: src && src.constructor && src.constructor.name,
        srcSize: src instanceof Blob ? src.size : (typeof src === 'string' ? src.length : 'N/A'),
        srcName: src instanceof File ? src.name : 'N/A'
    });
    
    if(!dropArea) {
        console.warn('[showInputPreview] dropArea not found, aborting');
        return;
    }
    
    const existing = dropArea.querySelector('.input-preview');
    if(existing) {
        console.log('[showInputPreview] removing existing preview');
        const objUrl = existing.dataset.objectUrl;
        if (objUrl) {
            URL.revokeObjectURL(objUrl);
            delete existing.dataset.objectUrl;
            console.log('[showInputPreview] revoked objectURL:', objUrl.slice(0, 50));
        }
        existing.remove();
    }

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
    console.log('[showInputPreview] img element inserted into DOM');

    const trySet = (url)=>{
        return new Promise((resolve)=>{
            img.onload = ()=>{
                img.style.visibility = 'visible';
                resolve(true);
            };
            img.onerror = (e)=>{
                // SILENT: caller will handle fallback next
                resolve(false);
            };
            try {
                img.src = url;
            } catch(e){
                resolve(false);
            }
        });
    };

    (async ()=>{
        try {
            // 1) If File/Blob and it's TIFF -> ask server to convert & return PNG dataURL
            if (src instanceof File || src instanceof Blob) {
                console.log('[showInputPreview] src is File/Blob, checking TIFF');
                const t = (src.type || '').toLowerCase();
                const name = (src.name || '').toLowerCase();
                const isTiff = t === 'image/tiff' || /\.tiff?$/i.test(name);
                console.log('[showInputPreview] isTiff:', isTiff, 'type:', t, 'name:', name);
                
                if (isTiff) {
                    console.log('[showInputPreview] uploading TIFF to server for preview...');
                    resultText.textContent = 'Đang tạo preview từ server...';
                    try {
                        const resp = await uploadFileToServer(src, 'gray8');
                        console.log('[showInputPreview] server TIFF response:', resp && Object.keys(resp));
                        if (resp && resp.dataURL) {
                            console.log('[showInputPreview] trying server dataURL, length:', resp.dataURL.length);
                            const ok = await trySet(resp.dataURL);
                            if (ok) {
                                console.log('[showInputPreview] ✅ TIFF preview success via server');
                                return;
                            } else {
                                console.warn('[showInputPreview] ❌ TIFF preview failed even with server dataURL');
                            }
                        } else {
                            console.warn('[showInputPreview] server response missing dataURL');
                        }
                    } catch (err) {
                        console.error('[showInputPreview] server TIFF preview error:', err);
                    }
                    console.log('[showInputPreview] TIFF preview failed, removing img');
                    img.remove();
                    return;
                }
                
                // not TIFF: try objectURL
                try {
                    const obj = URL.createObjectURL(src);
                    const ok = await trySet(obj);
                    if (ok) {
                        img.dataset.objectUrl = obj;
                        return;
                    } else {
                        // SILENT: will try server fallback next
                        URL.revokeObjectURL(obj);
                    }
                } catch(e){
                    // SILENT: will try server fallback
                }
                
                // FALLBACK: try uploading to server for preview (robust decode)
                try {
                    const resp = await uploadFileToServer(src, 'gray8');
                    if (resp && resp.dataURL) {
                        const ok2 = await trySet(resp.dataURL);
                        if (ok2) {
                            console.log('[showInputPreview] ✅ server preview success');
                            return;
                        }
                    }
                } catch (serverErr) {
                    console.warn('[showInputPreview] all preview attempts failed:', serverErr);
                }
            }

            // 2) If string dataURL (or other string) -> try directly
            if (typeof src === 'string') {
                const ok = await trySet(src);
                if (ok) {
                    console.log('[showInputPreview] ✅ string src preview success');
                    return;
                }
                console.warn('[showInputPreview] string src trySet failed');

                // if it was a data:image/tiff and failed, upload original file if available
                if (lastOriginalFile) {
                    try {
                        const resp = await uploadFileToServer(lastOriginalFile, 'gray8');
                        if (resp && resp.dataURL) {
                            const ok2 = await trySet(resp.dataURL);
                            if (ok2) {
                                console.log('[showInputPreview] ✅ server fallback success');
                                return;
                            }
                        }
                    } catch(e){
                        // silent — will remove preview below
                    }
                }
            }

            // fallback: remove preview img
            // silent fail — no need to log if preview simply not available
            img.remove();
        } catch (outerError) {
            console.error('[showInputPreview] outer catch error:', outerError);
            img.remove();
        }
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
    img.onload = ()=> {
        console.log('[showPreview] image loaded successfully');
    };
    img.onerror = (e)=> {
        // Only log if src is not being updated (avoid double error for intermediate preview)
        if (img.src === dataURL) {
            console.warn('[showPreview] failed to load final image', 'src preview:', (dataURL && (dataURL.slice ? dataURL.slice(0,80) : String(dataURL))));
        }
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

// Helper: collect extra params for mode (gamma/piecewise manual override)
function collectExtrasForMode(mode) {
    const extras = {};
    
    if (mode === 'gamma') {
        if (gammaInput && gammaInput.value) {
            extras.gamma = parseFloat(gammaInput.value);
        }
    } else if (mode === 'piecewise') {
        if (r1Input && r1Input.value) extras.r1 = parseInt(r1Input.value, 10);
        if (s1Input && s1Input.value) extras.s1 = parseInt(s1Input.value, 10);
        if (r2Input && r2Input.value) extras.r2 = parseInt(r2Input.value, 10);
        if (s2Input && s2Input.value) extras.s2 = parseInt(s2Input.value, 10);
    } else if (mode === 'denoise_manual') {
        // Collect denoise method and kernel
        if (denoiseMethodSelect && denoiseMethodSelect.value) {
            extras.method = denoiseMethodSelect.value;
        }
        if (denoiseKernelInput && denoiseKernelInput.value) {
            extras.kernel = parseInt(denoiseKernelInput.value, 10);
        }
    } else if (mode === 'sharpening') {
        // Collect sharpening type and amount
        if (sharpeningTypeSelect && sharpeningTypeSelect.value) {
            extras.sharp_type = sharpeningTypeSelect.value;
        }
        
    } else if (mode === 'frequency_filter') {
        // Collect filter type, cutoff, order
        if (freqFilterTypeSelect && freqFilterTypeSelect.value) {
            extras.filter_type = freqFilterTypeSelect.value;
        }
        if (freqCutoffInput && freqCutoffInput.value) {
            extras.cutoff = parseFloat(freqCutoffInput.value);
        }
        if (freqOrderInput && freqOrderInput.value) {
            extras.order = parseInt(freqOrderInput.value, 10);
        }
    } else if (mode === 'threshold') {
        // Collect threshold T
        if (thresholdInput && thresholdInput.value) {
            extras.threshold = parseInt(thresholdInput.value, 10);
        }
    } else if (mode === 'count_objects') {
        // Collect target class
        if (countClassInput && countClassInput.value) {
            extras.target_class = 'all';
        }
    }
    
    console.log('[collectExtrasForMode] mode:', mode, 'extras:', extras);
    return extras;
}