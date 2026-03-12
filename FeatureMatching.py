import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# テンプレートの読み込みと特徴点の検出
template = cv2.imread("template.JPEG", 0)
template = cv2.resize(template, None, fx=0.08, fy=0.08)
# orb = cv2.ORB_create(nfeatures=2000)
sift = cv2.SIFT_create()
# kp1, des1 = orb.detectAndCompute(template, None)
kp1, des1 = sift.detectAndCompute(template, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
h, w = template.shape

while True:
    if cv2.waitKey(1) == 27: # ESCキーで終了
        break

    # フレームの読み込みと特徴点の検出
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # kp2, des2 = orb.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(gray, None)
    if des2 is None: 
        continue

    # 特徴点のマッチング
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    matches = bf.knnMatch(des1, des2, k=2)
    good = matches[:20]

    # Ratio Test で良いマッチングを抽出
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    # ホモグラフィ変換の計算と描画
    if len(good) >= 10:
        # マッチングした特徴点の座標をそれぞれ抽出
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # ホモグラフィ行列を計算 (RANSACアルゴリズムで外れ値を除外)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 行列が正常に計算された場合のみ描画処理を行う
        if M is not None:
            # 1. テンプレート画像の四隅の座標を定義
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
            # 2. 四隅の座標をホモグラフィ行列でカメラ映像上の座標に変換
            dst = cv2.perspectiveTransform(pts, M)

            # 3. 変換された座標を使って、カメラフレームに枠線を描画 (緑色, 太さ3)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # カメラ映像のみを表示（drawMatchesによる並べる処理を削除）
    cv2.imshow("Object Tracking", frame)

cap.release()
cv2.destroyAllWindows()