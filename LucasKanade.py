import cv2
import numpy as np

cap = cv2.VideoCapture(0)

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

# 等間隔グリッドの生成（これは不変のベースとして保持する）
step = 20
points = []

for y in range(0, h, step):
    for x in range(0, w, step):
        points.append([x, y])

# base_p0 として初期グリッドを保存しておく
base_p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

while True:
    if cv2.waitKey(1) == 27: # ESCキーで終了
        break

    ret, frame = cap.read()
    if not ret:
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 毎フレーム、必ず固定のグリッド(base_p0)を探索の起点にする
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, base_p0, None, **lk_params)

    # st==1 はオプティカルフローが見つかった点
    good_new = p1[st==1]
    good_old = base_p0[st==1]

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        dx = a - c
        dy = b - d

        # 動きが大きい場合だけ描画
        if np.sqrt(dx*dx + dy*dy) > 2:
            cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

    cv2.imshow("Grid Optical Flow", frame)

    # 次のフレーム比較のために画像を更新するが、
    # 探索点は base_p0 を使い続けるため p0 の更新は行わない
    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()