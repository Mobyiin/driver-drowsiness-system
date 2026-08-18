import cv2

def render_ui(frame, is_fatigued, alert_triggered, is_drowsy_active, drowsy_duration, capture_fps=0.0, inference_fps=0.0, target_bbox=None):

    if is_fatigued:
        text = "CRITICAL: TAKE A BREAK NOW!"
        color = (0, 0, 255)
    elif alert_triggered:
        text = f"DANGER! DROWSY ({drowsy_duration:.1f}s)"
        color = (0, 0, 255)
    elif is_drowsy_active:
        text = f"Warning: Drowsy ({drowsy_duration:.1f}s)"
        color = (0, 255, 255)
    else:
        text = "Status: Awake"
        color = (0, 255, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    x, y = 30, 50


    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        box_color = (0, 0, 255) if (is_drowsy_active or is_fatigued) else (0, 255, 0)
        


    height, width, _ = frame.shape
    fps_text = f"Cam FPS: {capture_fps:.1f} | YOLO FPS: {inference_fps:.1f}"
    fps_font_scale = 0.5
    fps_thickness = 1
    (fps_w, fps_h), fps_baseline = cv2.getTextSize(fps_text, font, fps_font_scale, fps_thickness)
    fps_x = width - fps_w - 20
    fps_y = 35

    cv2.rectangle(frame, (fps_x - 8, fps_y - fps_h - 8), (fps_x + fps_w + 8, fps_y + fps_baseline + 4), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (fps_x, fps_y), font, fps_font_scale, (0, 255, 0), fps_thickness, cv2.LINE_AA)

    return frame
