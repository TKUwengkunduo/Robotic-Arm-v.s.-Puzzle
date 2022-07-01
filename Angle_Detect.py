from multiprocessing.connection import wait
import pyrealsense2 as rs           # 版本目前不支援python3.10
from cmath import sqrt
import numpy as np
import cv2
import math
import time

# video = cv2.VideoCapture("/home/weng/ICLAB/output2.avi")

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out_video = cv2.VideoWriter('demo_Angle_Detect_canny.avi', fourcc, 30.0, (1280,  720))


'''
    用完後必關！！！！！！！！重要！！！！！
'''
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# pipeline.start(config)
# sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
# sensor.set_option(rs.option.auto_exposure_priority, True)




"""
    檢測
        輸入: 圖片、k_gauss(高斯模糊)、low_threshold(canny低閥值)、high_threshold(canny高閥值)、k_close(閉運算)
        輸出: 拼圖角度、拼圖座標
"""
def detect(img, k_gauss=17, low_threshold=10, high_threshold=20, k_close=3):

    canny_img = Canny(img, k_gauss, low_threshold, high_threshold, k_close)
    contour_img, box = Fined_Contour(img, canny_img)
    detect_img, center, angle = detect_angle(contour_img, box)
    
    cv2.imshow('detect_img', detect_img)
    # cv2.imshow('canny_img', canny_img)
    cv2.waitKey(3000)


    return angle, center


"""
    檢測全部拼圖
        輸入: 圖片、k_gauss(高斯模糊)、low_threshold(canny低閥值)、high_threshold(canny高閥值)、k_close(閉運算)
        輸出: 拼圖角度、拼圖座標
"""
def detect_ALL(img, k_gauss=17, low_threshold=10, high_threshold=20, k_close=3):

    canny_img = Canny(img, k_gauss, low_threshold, high_threshold, k_close)
    contour_img, cv2_info = Fined_Contour_ALL(img, canny_img)
    
    cv2.imshow('canny_img', canny_img)
    cv2.imshow('contour_img', contour_img)
    cv2.waitKey(3000)


    return cv2_info






"""
    Canny演算法
        輸入: (原始圖片、k_gauss(高斯模糊)、low_threshold(canny低閥值)、high_threshold(canny高閥值)、k_close(閉運算))
        輸出: 邊緣檢測圖片

        註記:
        1. 高斯模糊程度越小邊緣檢測結果越符合原圖、但容易有雜訊。
        2. 閉運算可有效解決canny斷線問題，但太大一樣會有第一點問題。
"""
def Canny(orig_img, k_gauss, low_threshold, high_threshold, k_close):
    # 灰階
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    gauss_img = cv2.GaussianBlur(orig_img, (k_gauss, k_gauss), 0)

    # Canny邊緣檢測
    canny_img = cv2.Canny(gauss_img, low_threshold, high_threshold)

    # 閉運算(緩解Canny斷線問題)
    kernel = np.ones((k_close,k_close),np.uint8)
    gradient = cv2.morphologyEx(canny_img, cv2.MORPH_GRADIENT, kernel)

    return gradient



"""
    尋找輪廓
        輸入: (原始圖片、邊緣檢測圖片)
        輸出: (輪廓圖片、邊框四點座標)
"""
def Fined_Contour(orig_img, canny_img):

    #複製原圖片
    Contour_img = orig_img.copy()

    # 輪廓檢測(使用Canny檢測後影像繼續檢測)
    contours,hierarchy = cv2.findContours(canny_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # 尋找最大輪廓
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    cnt = contours[max_idx]

    # 繪製輪廓
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)              # 轉換為整數
    cv2.drawContours(Contour_img, [box], 0, (0, 255, 255), 1)

    return Contour_img,box



    

"""
    檢測角度
        輸入: (輪廓圖片、邊框四點座標)
        輸出: 檢測圖片、中心點座標、角度
"""
def detect_angle(Contour_img,box):
    # 尋找長邊、短邊
    tem_length_1to2 = sqrt(pow(box[0][0]-box[1][0],2) + pow(box[0][1]-box[1][1],2)).real
    tem_length_1to4 = sqrt(pow(box[0][0]-box[3][0],2) + pow(box[0][1]-box[3][1],2)).real
    if tem_length_1to2 > tem_length_1to4:
        star = (0,3)
        end = (1,2)
    else:
        star = (0,1)
        end = (2,3)
    x_start = ( (box[star[0]][0]+box[star[1]][0])/2 )
    y_start = ( (box[star[0]][1]+box[star[1]][1])/2 )
    x_end = ( (box[end[0]][0]+box[end[1]][0])/2 )
    y_end = ( (box[end[0]][1]+box[end[1]][1])/2 )

    # 分辨兩個角度(暫時)
    if y_start>y_end:
        # 左上角(0,0)
        tem = x_start
        x_start = x_end
        x_end = tem
        tem = y_start
        y_start = y_end
        y_end = tem
    else:
        pass

    # 繪製中心線
    cv2.line(Contour_img, (int(x_start),int(y_start)), (int(x_end),int(y_end)), (0, 100, 255), 1)
    
    # 計算角度(以正x為0，順實為-,逆實為+)
    angle_1 = round(math.degrees(math.atan2( -(y_start-((y_start+y_end)/2)) , x_start-((x_start+x_end)/2) )), 5)
    angle_2 = round(math.degrees(math.atan2( -(y_end - ((y_start+y_end)/2)) , x_end - ((x_start+x_end)/2) )), 5)

    # 繪製中心點
    cv2.circle(Contour_img,( (int((x_start+x_end)/2),int((y_start+y_end)/2)) ),5,[0,0,255],1)

    # 繪製角度
    # cv2.arrowedLine(Contour_img, (int((x_start+x_end)/2),int((y_start+y_end)/2)), (int(x_start),int(y_start)), (255, 0, 0), 3)
    # cv2.arrowedLine(Contour_img, (int((x_start+x_end)/2),int((y_start+y_end)/2)), (int(x_end),int(y_end)), (0, 0, 255), 3)
    # cv2.putText(Contour_img,str(angle_1),(int(x_start),int(y_start)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 20, 0), 2, cv2.LINE_AA)
    # cv2.putText(Contour_img,str(angle_2),(int(x_end),int(y_end)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 20, 100), 2, cv2.LINE_AA)
    cv2.putText(Contour_img,'Center='+'('+str((x_start+x_end)/2)+','+str((y_start+y_end)/2)+')',(0,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(Contour_img,'Angel='+str(angle_1),(0,70),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 2, cv2.LINE_AA)
    cv2.putText(Contour_img,'Angel='+str(angle_2),(0,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 2, cv2.LINE_AA)
    
    # 繪製參考線(0度)
    #cv2.arrowedLine(Contour_img, (int((x_start+x_end)/2),int((y_start+y_end)/2)), (int((x_start+x_end)/2)+45,int((y_start+y_end)/2)), (0, 0, 0), 2)
    #cv2.line(Contour_img, (int((x_start+x_end)/2),int((y_start+y_end)/2)), (int((x_start+x_end)/2)-45,int((y_start+y_end)/2)), (0, 0, 0), 2)


    return Contour_img, [ (x_start+x_end)/2 , (y_start+y_end)/2 ], [angle_1,angle_2]







def Fined_Contour_ALL(orig_img, canny_img):
    cnt = []
    cv2_info=[]     #[[X、Y], 角度, 大小, box, 是否有重複框]*n

    #複製原圖片
    Contour_img = orig_img.copy()

    # 輪廓檢測(使用Canny檢測後影像繼續檢測)
    contours,hierarchy = cv2.findContours(canny_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # 尋找所有輪廓
    for k in range(len(contours)):
        # 第一層大小過濾
        # if cv2.contourArea(contours[k])<18000 and cv2.contourArea(contours[k])>13000:
        #     cnt.append(contours[k])
        if cv2.contourArea(contours[k])<18000 and cv2.contourArea(contours[k])>12000:
            cnt.append(contours[k])

    # 計算所有中心位置與角度
    for k in range(len(cnt)):
        rect = cv2.minAreaRect(cnt[k])
        box = cv2.boxPoints(rect)
        box = np.int0(box)              # 轉換為整數
        center, angle, size= find_center(box)
        cv2_info.append([center, angle, size, box, 0])
        cv2.drawContours(Contour_img, [box], 0, (5, 150, 5), 1)
        # 繪製中心點
        cv2.circle(Contour_img,(int(center[0]),int(center[1])),5,[0,0,255],1)
    # cv2.imshow('Contour_img',Contour_img)
    # cv2.waitKey(0)
    # for i in range(len(cv2_info)):
    #     print(i,"= ",cv2_info[i])


    
    # 第二層排序並過濾重複框
    k=0
    while k<len(cv2_info):
        k+=1
        if k>=len(cv2_info):continue
        for l in range(len(cv2_info)):
            if l>=len(cv2_info):continue

            temp = sqrt( pow(cv2_info[k][0][0]-cv2_info[l][0][0],2) + pow(cv2_info[k][0][1]-cv2_info[l][0][1],2) ).real
            if temp<1 and (k!=l):
                # print(k,l)
                # k-=1
                if abs(cv2_info[k][2]-cv2_info[l][2])>1800:
                    if cv2_info[k][2]>cv2_info[l][2]:
                        del cv2_info[k]
                        if k>l:
                            cv2_info[l][4]=1
                        else:
                            cv2_info[k][4]=1
                    else:
                        del cv2_info[l]
                        if k>l:
                            cv2_info[l][4]=1
                        else:
                            cv2_info[k][4]=1
                    break
                else:
                    if cv2_info[k][2]>cv2_info[l][2]:
                        del cv2_info[l]
                        if k>l:
                            cv2_info[l][4]=1
                        else:
                            cv2_info[k][4]=1
                    else:
                        del cv2_info[k]
                        if k>l:
                            cv2_info[l][4]=1
                        else:
                            cv2_info[k][4]=1
                    break
            
        

    for k in range(len(cv2_info)):
        # 繪製輪廓
        rect = cv2.minAreaRect(cv2_info[k][3])
        box = cv2.boxPoints(rect)
        box = np.int0(box)              # 轉換為整數
        cv2.drawContours(Contour_img, [box], 0, (0, 255, 255), 1)

    # print(cv2_info)


    return Contour_img,cv2_info


def find_center(box):
    # 尋找長邊、短邊
    tem_length_1to2 = sqrt(pow(box[0][0]-box[1][0],2) + pow(box[0][1]-box[1][1],2)).real
    tem_length_1to4 = sqrt(pow(box[0][0]-box[3][0],2) + pow(box[0][1]-box[3][1],2)).real
    if tem_length_1to2 > tem_length_1to4:
        star = (0,3)
        end = (1,2)
    else:
        star = (0,1)
        end = (2,3)
    x_start = ( (box[star[0]][0]+box[star[1]][0])/2 )
    y_start = ( (box[star[0]][1]+box[star[1]][1])/2 )
    x_end = ( (box[end[0]][0]+box[end[1]][0])/2 )
    y_end = ( (box[end[0]][1]+box[end[1]][1])/2 )

    # 分辨兩個角度(暫時)
    if y_start>y_end:
        # 左上角(0,0)
        tem = x_start
        x_start = x_end
        x_end = tem
        tem = y_start
        y_start = y_end
        y_end = tem
    else:
        pass

    angle_1 = round(math.degrees(math.atan2( -(y_start-((y_start+y_end)/2)) , x_start-((x_start+x_end)/2) )), 5)
    angle_2 = round(math.degrees(math.atan2( -(y_end - ((y_start+y_end)/2)) , x_end - ((x_start+x_end)/2) )), 5)

    return [ (x_start+x_end)/2 , (y_start+y_end)/2 ], [angle_1,angle_2],round(tem_length_1to2*tem_length_1to4,3)




def Draw_angle_ALL(Contour_img,cv2_info):

    for i in range(len(cv2_info)):
        # 繪製中心點
        cv2.circle(Contour_img,( int(cv2_info[i][0][0]), int(cv2_info[i][0][1])) ,5,[0,0,255],1)

        cv2.putText(Contour_img,
                    'Center='+'('+str(cv2_info[i][0][0])+','+str(cv2_info[i][0][1])+')',
                    (int(cv2_info[i][0][0]-60),int(cv2_info[i][0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(Contour_img,'Angel='+str(cv2_info[i][1])+str(cv2_info[i][1]),(int(cv2_info[i][0][0]-60),int(cv2_info[i][0][1]+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 20, 255), 1, cv2.LINE_AA)
        # cv2.putText(Contour_img,'Angel='+str(cv2_info[i][1]),(0,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 2, cv2.LINE_AA)

    # print(cv2_info)
    


    return Contour_img







"""
顯示圖片
    輸入: (欲顯示圖片、長、寬)
    輸出: 無
"""
def show_img(name,img):
    # cv2.namedWindow( name, cv2.WINDOW_NORMAL )
    # cv2.resizeWindow(name, 640, 360)
    cv2.imshow(name, img)
    cv2.waitKey(5)





if __name__=="__main__":

    # ret, image = video.read()

    # 讀取影像
    img = cv2.imread('/home/weng/Downloads/puzzle_0021.jpg')
# orig_img = cv2.resize(orig_img, (640 ,360), interpolation=cv2.INTER_AREA)

    # frames = pipeline.wait_for_frames()
    # img = frames.get_color_frame()
    # img = np.asanyarray(img.get_data())
    # time.sleep(1)
    # frames = pipeline.wait_for_frames()
    # img = frames.get_color_frame()
    # img = np.asanyarray(img.get_data())

    img = cv2.imread('/home/weng/Downloads/16339125841567.jpg')

    '''
        單片檢測
    '''
    # canny_img = Canny(img, k_gauss=17, low_threshold=10, high_threshold=20, k_close=3)
    # contour_img, box = Fined_Contour(img, canny_img)
    # detect_img, center, angle = detect_angle(contour_img, box)
    # show_img('orig_img', img)
    # show_img('canny_img', canny_img)
    # show_img('contour_img', contour_img)
    # show_img('detect_img', detect_img)
    # cv2.waitKey(0)
    # print('center= ',center)
    # print('angle= ',angle)


    '''
        多片檢測
    '''
    # canny_img = Canny(img, k_gauss=17, low_threshold=10, high_threshold=20, k_close=3)
    # contour_img, cv2_info = Fined_Contour_ALL(img, canny_img)
    # # detect_img = Draw_angle_ALL(contour_img,cv2_info)
    # show_img('orig_img', img)
    # show_img('canny_img', canny_img)
    # show_img('contour_img', contour_img)
    # cv2.waitKey(0)
    cv2_info = detect_ALL(img, k_gauss=13, low_threshold=10, high_threshold=20, k_close=4)
    for i in range(len(cv2_info)):
        print(i,"=",cv2_info[i])
    # print(cv2_info)
    cv2.waitKey(0)


# video.release()
# out_video.release()
cv2.destroyAllWindows()


