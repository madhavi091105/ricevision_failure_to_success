cd C:\Users\madha\OneDrive\Desktop\train_yolo

yolo detect predict model=C:\Users\madha\OneDrive\Desktop\best_model.pt source="C:\Users\madha\Downloads\rr12.jpeg" save=True save_txt=True

python -c "import cv2; img=cv2.imread(r'C:\Users\madha\Downloads\rr12.jpeg'); h,w=img.shape[:2]; lines=open(r'runs\detect\predict2\labels\rr12.txt').read().strip().split('\n'); import os; os.makedirs('crops',exist_ok=True); 
for i,l in enumerate(lines):
 c=l.split(); x,y,bw,bh=map(float,c[1:5]); x*=w; y*=h; bw*=w; bh*=h; x1=int(x-bw/2); y1=int(y-bh/2); x2=int(x+bw/2); y2=int(y+bh/2); crop=img[y1:y2,x1:x2]; cv2.imwrite(f'crops/crop_{i}.png',crop)"