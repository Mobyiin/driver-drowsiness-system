import cv2

def render_ui(frame, is_fatigued, alert_triggered, is_drowsy_active, drowsy_duration):
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
    cv2.rectangle(
        frame, 
        (x - 10, y - text_height - 10), 
        (x + text_width + 10, y + baseline + 5), 
        (0, 0, 0), 
        -1
    )

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame