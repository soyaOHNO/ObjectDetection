import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 背景差分器 (varThresholdを少し上げて微小なノイズを無視しやすくする)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

lk_params = dict(
    winSize=(15,15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

ret, old_frame = cap.read()
if not ret:
    exit()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

h, w = old_gray.shape

step = 20
points = []
for y in range(0, h, step):
    for x in range(0, w, step):
        points.append([x, y])

base_p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# モルフォロジー変換用のカーネル（少し大きめにして穴をしっかり埋める）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    if cv2.waitKey(1) == 27: # ESCキーで終了
        break

    ret, frame = cap.read()
    if not ret:
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 【改善1】前処理：ガウシアンブラーでカメラの細かいザラザラノイズを潰す
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # 背景差分を適用
    fgmask = fgbg.apply(frame_blur)
    
    # 【改善2】影の除去：MOG2は影をグレー(127)で出すので、白(255)だけを抽出
    _, fgmask_clean = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # 【改善3】後処理（モルフォロジー変換のコンボ）
    # 1. オープニング：外側に散らばった細かい白点（ごみ）を消す
    fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_OPEN, kernel)
    # 2. クロージング：物体内部の黒い穴（スカスカ部分）を埋める
    fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_CLOSE, kernel)
    # 3. 膨張：マスクを少し太らせて、物体のエッジ付近のフローも確実に拾えるようにする
    fgmask_clean = cv2.dilate(fgmask_clean, kernel, iterations=2)

    # オプティカルフローの計算
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, base_p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = base_p0[st==1]

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        
        ix, iy = int(c), int(d)
        
        if 0 <= iy < h and 0 <= ix < w:
            # 強化されたマスクを使って判定
            if fgmask_clean[iy, ix] == 255:
                dx = a - c
                dy = b - d

                if np.sqrt(dx*dx + dy*dy) > 1.0:
                    cv2.arrowedLine(frame, (ix, iy), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                    cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

    cv2.imshow("Grid + Enhanced Mask", frame)
    cv2.imshow("Foreground Mask (Cleaned)", fgmask_clean)

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()