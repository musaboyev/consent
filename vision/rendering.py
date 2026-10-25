"""Protection is rendered into independent image pixels, never a UI overlay."""
import cv2
import numpy as np

def head_box(box):
    x, y, w, h = box
    # Full facial area with modest padding; keep hair/head silhouette intact.
    return (x - .08*w, y - .06*h, w*1.16, h*1.14)

def protect_scene(image):
    """Privacy uncertainty fallback: live scene context, no recoverable face detail.

    This is only used after successful processing with uncertain locations.
    Actual processing/camera failures still use the opaque error slate.
    """
    height,width=image.shape[:2]
    coarse=cv2.resize(image,(12,9),interpolation=cv2.INTER_AREA)
    obscured=cv2.resize(coarse,(width,height),interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(obscured,(0,0),8)

def protect(image, boxes=(), shield=False):
    out = image.copy()
    if shield:
        out[:] = (28, 23, 20)
        cv2.putText(out, "PROTECTED OUTPUT SHIELDED", (18, out.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, .62, (190, 224, 209), 1, cv2.LINE_AA)
        return out
    height, width = out.shape[:2]
    for box in boxes:
        x, y, w, h = box
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x+w+1)), min(height, int(y+h+1))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        # Blur the actual photograph. Preserve low-frequency lighting/face shape
        # instead of replacing the head with a few averaged colors.
        obscured = cv2.GaussianBlur(roi, (0, 0), max(10.,min(w,h)*.13))
        mask = np.zeros((y1-y0,x1-x0),np.uint8)
        # Use unclipped geometry so heads at image edges keep full coverage.
        center=(round(x+w/2-x0),round(y+h/2-y0))
        axes=(max(1,round(w/2)),max(1,round(h/2)))
        cv2.ellipse(mask,center,axes,0,0,360,255,-1,cv2.LINE_AA)
        feather=max(3.,min(w,h)*.045)
        mask=cv2.GaussianBlur(mask,(0,0),feather,borderType=cv2.BORDER_REPLICATE)
        alpha=mask.astype(np.float32)[...,None]/255.
        out[y0:y1,x0:x1]=(roi*(1-alpha)+obscured*alpha).astype(np.uint8)
    return out
