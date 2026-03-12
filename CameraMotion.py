import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, old_frame = cap.read()
if not ret:
    exit()

h, w = old_frame.shape[:2]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

focal_length = w
center = (w / 2, h / 2)
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]], dtype=np.float64)

feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=3)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def get_combined_features(gray_img):
    corners = cv2.goodFeaturesToTrack(gray_img, mask=None, **feature_params)
    step = 40
    grid_points = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            grid_points.append([[np.float32(x), np.float32(y)]])
    grid_points = np.array(grid_points, dtype=np.float32)
    
    if corners is not None:
        combined = np.vstack((corners, grid_points))
    else:
        combined = grid_points
    return combined

p0 = get_combined_features(old_gray)

t_f = np.zeros((3, 1))
R_f = np.eye(3)
traj_img = np.zeros((600, 600, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 20:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # 1. まず追跡に成功した点だけを抽出
        good_new_temp = p1[st == 1]
        good_old_temp = p0[st == 1]

        good_new = []
        good_old = []

        # 2. 【改善】微小な動き（ノイズ）を個別に弾く
        for new, old in zip(good_new_temp, good_old_temp):
            dx = new[0] - old[0]
            dy = new[1] - old[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # 1.0ピクセル以上動いた点だけを採用（それ以下はカメラノイズの揺れとみなす）
            if dist > 1.0:
                good_new.append(new)
                good_old.append(old)

        good_new = np.array(good_new, dtype=np.float32)
        good_old = np.array(good_old, dtype=np.float32)

        # 動いた点が十分に生き残っている場合のみ計算
        if len(good_new) > 8:
            E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None and E.shape == (3, 3):
                _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, K)

                # 3. 【改善】カメラ全体の移動量が極端に小さい場合は無視する
                # tは単位ベクトルなので、実際にどのくらい動いたかを「インライアの平均ピクセル移動距離」で測る
                inlier_dists = []
                for i in range(len(good_new)):
                    if mask_pose is not None and mask_pose[i][0] == 255:
                        dx = good_new[i][0] - good_old[i][0]
                        dy = good_new[i][1] - good_old[i][1]
                        inlier_dists.append(np.sqrt(dx*dx + dy*dy))
                
                # インライアの平均移動距離が2.0ピクセルより大きい場合のみ、カメラが動いたと判定
                if len(inlier_dists) > 0 and np.mean(inlier_dists) > 2.0:
                    t_f = t_f + R_f.dot(t)
                    R_f = R.dot(R_f)

                draw_x = int(t_f[0][0] * 10) + 300
                draw_z = int(t_f[2][0] * 10) + 300
                draw_x = max(0, min(600 - 1, draw_x))
                draw_z = max(0, min(600 - 1, draw_z))

                cv2.circle(traj_img, (draw_x, draw_z), 1, (0, 255, 0), 2)
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    if mask_pose is not None and mask_pose[i][0] == 255:
                        a, b = new.ravel()
                        cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        if len(good_new) < 150:
            p0 = get_combined_features(frame_gray)
        else:
            p0 = good_new.reshape(-1, 1, 2)
    else:
        p0 = get_combined_features(frame_gray)

    cv2.imshow("Hybrid Features View", frame)
    cv2.imshow("3D Trajectory", traj_img)

    if cv2.waitKey(1) == 27:
        break
        
    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()