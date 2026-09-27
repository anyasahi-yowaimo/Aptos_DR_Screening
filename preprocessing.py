#CELL 11 — Deterministic preprocessing pipeline for fundus images
# (same transform used for train/val/test; only used to prepare a clean base image,
#  before Albumentations augmentation is applied on top in Cell 12/13)

def circular_crop(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def ben_graham_normalize(img, sigma_frac=10):
    sigma = img.shape[1] / sigma_frac
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def circular_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def preprocess_image(path, size=CFG["IMG_SIZE"]):
    img = cv2.imread(path)
    img = circular_crop(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = circular_mask(img)
    img = ben_graham_normalize(img)
    img = apply_clahe(img)
    return img

def process_and_cache_split(split_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Preprocessing -> {out_dir}"):
        out_path = os.path.join(out_dir, row["id_code"] + ".png")
        if os.path.exists(out_path):
            continue
        processed = preprocess_image(row["image_path"])
        cv2.imwrite(out_path, processed)

process_and_cache_split(train_df, os.path.join(CFG["PROCESSED_DIR"], "train_split"))
process_and_cache_split(val_df, os.path.join(CFG["PROCESSED_DIR"], "val_split"))
process_and_cache_split(test_df, os.path.join(CFG["PROCESSED_DIR"], "test_split"))
print("Preprocessing complete. Cached processed images on disk.")