import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# 動画のFPSを取得（カメラの場合は手動設定が必要な場合あり）
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps > 100: fps = 30  # 取得失敗時のデフォルト

# カメラの解像度を取得（タイマー配列を作るため）
ret, test_frame = cap.read()
if not ret:
    print("カメラから映像を取得できませんでした。")
    exit()
height, width = test_frame.shape[:2]

# --- 追加：静止タイマーの初期化 ---
# 各ピクセルが「何フレーム連続で前景か」をカウントする配列
stationary_timer = np.zeros((height, width), dtype=np.int32)
# 何秒静止したら背景とみなすかの閾値（例: 3秒）
stationary_threshold_frames = int(fps * 3.0) 


# 1 背景生成
def get_initial_background(cap, fps, duration=1.0):
    print(f"{duration}秒分のフレームを読み込んで背景を生成中...")
    frames = []
    frame_count = int(fps * duration)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float64))
    if not frames:
        return None

    avg_background = np.mean(frames, axis=0)
    return avg_background.astype(np.uint8)


background = get_initial_background(cap, fps, duration=1.0)

if background is not None:
    cv2.imshow("Background", background)
    print("背景生成完了")

    while True:
        if cv2.waitKey(1) == 27: # ESCキーで終了
            break
        ret, frame = cap.read()
        if not ret: break

        # 2 物体検出
        diff = cv2.absdiff(frame, background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 3 マスク修正
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # --- 修正: 4 背景更新（np.whereを使用して安全に合成） ---
        # 画面全体で一旦ブレンド画像を計算しておく
        blended = cv2.addWeighted(background, 0.9, frame, 0.1, 0)
        # maskを3チャンネルに拡張（BGR画像と形状を合わせるため）
        mask_3d = cv2.merge([mask, mask, mask])
        # maskが0（黒=背景）の場所はblended、それ以外は元のbackgroundを維持
        background = np.where(mask_3d == 0, blended, background)

        # --- 追加: 5 長期静止対応 ---
        # maskが255（前景）ならタイマー+1、0（背景）なら0にリセット
        stationary_timer = np.where(mask > 0, stationary_timer + 1, 0)
        
        # タイマーが閾値を超えたピクセルのTrue/Falseマップを作成
        force_bg_mask = stationary_timer > stationary_threshold_frames
        
        # Trueになった場所だけ、現在のフレームで背景を上書き（強制インジェクト）
        force_bg_mask_3d = np.broadcast_to(force_bg_mask[..., None], background.shape)
        background = np.where(force_bg_mask_3d, frame, background)
        
        # 背景に取り込んだピクセルのタイマーは用済みなのでリセット
        stationary_timer[force_bg_mask] = 0

        move_objects = cv2.bitwise_and(frame, frame, mask=mask)

        # 結果の表示
        cv2.imshow("Live", frame)
        cv2.imshow("Mask", mask)               # 検出結果の確認用
        cv2.imshow("Updating Background", background) # 背景がどう育つか確認用
        cv2.imshow("move_objects", move_objects) # 動いている物体だけの映像
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()