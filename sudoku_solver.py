import cv2
import numpy as np
# import pytesseract

# config=("-l eng --oem 1 --psm 10")

def find_longest_edge(pt_top_left,pt_top_right,pt_bottom_left,pt_bottom_right):
	maxlen=max([cal_distance(pt_top_left,pt_top_right),cal_distance(pt_top_left,pt_bottom_left),cal_distance(pt_bottom_left,pt_bottom_right),cal_distance(pt_bottom_right,pt_top_right)])
	return np.sqrt(maxlen)

def cal_intersection_points(top_edge,left_edge,bottom_edge,right_edge,width,height): ## function for calculating intersection point
## of two lines. source :- https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
	left1=[0,left_edge[0]/np.sin(left_edge[1])]
	left2=[width,left1[1]-width/np.tan(left_edge[1])]

	right1=[0,right_edge[0]/np.sin(right_edge[1])]
	right2=[width,right1[1]-width/np.tan(right_edge[1])]

	top1=[0,top_edge[0]/np.sin(top_edge[1])]
	top2=[width,top1[1]- width/np.tan(top_edge[1])]

	bottom1=[0,bottom_edge[0]/np.sin(bottom_edge[1])]
	bottom2=[width,bottom1[1] - width/np.tan(bottom_edge[1])]

	leftA=left2[1]-left1[1]
	leftB=left1[0]-left2[0]
	leftC=leftA*left1[0]+leftB*left1[1]

	rightA=right2[1]-right1[1]
	rightB=right1[0]-right2[0]
	rightC=rightA*right1[0]+rightB*right1[1]

	topA=top2[1]-top1[1]
	topB=top1[0]-top2[0]
	topC=topA*top1[0]+topB*top1[1]

	bottomA=bottom2[1]-bottom1[1]
	bottomB=bottom1[0]-bottom2[0]
	bottomC=bottomA*bottom1[0]+bottomB*bottom1[1]

	detTopLeft=leftA*topB-leftB*topA
	detTopRight=rightA*topB-rightB*topA
	detBottomLeft=leftA*bottomB-leftB*bottomA
	detBottomRight=rightA*bottomB-rightB*bottomA

	pt_top_left=[(topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft]
	pt_top_right=[(topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight]
	pt_bottom_left=[(bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft]
	pt_bottom_right=[(bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight]

	return [pt_top_left,pt_top_right,pt_bottom_left,pt_bottom_right] ## returns the four corners of the sudoku puzzle

def cal_distance(p1,p2):
	return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

def retain_largest_blob(img,height,width):
	mask = np.zeros((height+2, width+2), np.uint8)
	m=-1
	for x in range(height):
		for y in range(width):
			if(img[x][y]>=128):
				area=cv2.floodFill(img,mask,(y,x),64)
				
				if(area[0]>m):
					m=area[0]
					m_pt=(y,x)

	mask = np.zeros((height+2, width+2), np.uint8)
	area=cv2.floodFill(img,mask,m_pt,255)

	for x in range(height):
		for y in range(width):
			if(img[x][y]==64 and x!=m_pt[1] and y!=m_pt[0]): # floodfill all the connected components with black except the largest blob  
				area=cv2.floodFill(img,mask,(y,x),0) 

img_name='sudoku-original.jpg'

# for reference - horizontal direction is x-axis, vertical direction is y-axis in image
img =cv2.imread(img_name,0)


[height,width]=np.shape(img)
if(height>600 or width>600):
	print("Image size exceeded")
	img=cv2.resize(img,(600,600))
	height=600
	width=600

cv2.imshow('Original puzzle',img)

processed_img=cv2.GaussianBlur(img,(11,11),0) #blurring is done to smoothen out the noise


thresh1 = cv2.adaptiveThreshold(processed_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2) 
# threshold level is calculated for windows of size 5x5. The algo calculates the mean of the 5x5 grid, subtracts 2 from it and assigns it as an adaptive threshold for that grid
inverted_image=cv2.bitwise_not(thresh1)

kernel=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

dilated_img=cv2.dilate(inverted_image,kernel) # using a plus shaped kernel for dilation

#cv2.imshow('my image',dilated_img)
#cv2.waitKey(0)
 
retain_largest_blob(dilated_img,height,width)
dilated_img=cv2.erode(dilated_img,kernel);

edges=cv2.Canny(dilated_img,50,150,3)   # need to check this again
lines=cv2.HoughLines(edges,1,np.pi/180,100) # fourth argument is threshold for minimum vote for (rho,theta) pair to be recognized as a line
line_test=np.zeros((height,width))
num_lines=len(lines)
#print(num_lines)

##merging close lines

for i in range(num_lines):
	if(lines[i][0][0]==0 and lines[i][0][1]==-100): ## such lines are ignored
		continue

	rho1=lines[i][0][0]
	theta1=lines[i][0][1]
	pt1=[0,0]
	pt2=[0,0]
	if(theta1>np.pi*45/180 and theta1<np.pi*135/180):  ## theta is between 45 deg and 135 deg (case for horizontal lines)
		pt1[0]=0                            ## choosing two points on a horizontal line at extreme ends of the image
		pt1[1]=rho1/np.sin(theta1)
		pt2[0]=width
		pt2[1]=-width/np.tan(theta1)+rho1/np.sin(theta1)
	else:                                               ## case for vertical lines
		pt1[0]=rho1/np.cos(theta1)          ## choosing two points on a vertical line at extreme ends of the image
		pt1[1]=0
		pt2[0]=rho1/np.cos(theta1)-height*np.tan(theta1)
		pt2[1]=height

	for j in range(num_lines):
		if(i==j):
			continue
		## checking if two lines are nearby and oriented in almost same direction
		if( abs(lines[i][0][0]-lines[j][0][0])<20 and abs(lines[i][0][1]-lines[j][0][1])<np.pi*10/180 ): 
			pt1_current=[0,0]
			pt2_current=[0,0]
			rho2=lines[j][0][0]
			theta2=lines[j][0][1]

			if(theta2>np.pi*45/180 and theta2<np.pi*135/180):  ## theta is between 45 deg and 135 deg (for almost horizontal lines)
				pt1_current[0]=0
				pt1_current[1]=rho2/np.sin(theta2)
				pt2_current[0]=width
				pt2_current[1]=-width/np.tan(theta2)+rho2/np.sin(theta2)
			else:                                            ## case for vertical lines
				pt1_current[0]=rho2/np.cos(theta2)
				pt1_current[1]=0
				pt2_current[0]=rho2/np.cos(theta2)-height*np.tan(theta2)
				pt2_current[1]=height
            
            ## merging two nearby lines		
			if(cal_distance(pt1,pt1_current)<64*64 and cal_distance(pt2,pt2_current)<64*64): 
				lines[i][0][0]=(lines[i][0][0]+lines[j][0][0])/2;
				lines[i][0][1]=(lines[i][0][1]+lines[j][0][1])/2;

				lines[j][0][0]=0
				lines[j][0][1]=-100

for i in range(num_lines):     # routine for drawing the merged lines
	rho=lines[i][0][0]
	theta=lines[i][0][1]
	if(rho==0 and theta==-100):
		continue
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(line_test,(x1,y1),(x2,y2),255,1)

top_edge=[1000,1000]
bottom_edge=[-1000,-1000]
right_edge=[-1000,-1000]
left_edge=[1000,1000]
top_edge_x_icpt=0
top_edge_y_icpt=100000
left_edge_x_icpt=100000
left_edge_y_icpt=0
bottom_edge_x_icpt=0
bottom_edge_y_icpt=0
right_edge_x_icpt=0
right_edge_y_icpt=0

for i in range(num_lines):
	if(lines[i][0][0]==0 and lines[i][0][1]==-100): # ignoring the merged lines
		continue
	rho=lines[i][0][0]
	theta=lines[i][0][1]

	x_icpt=rho/np.cos(max(theta,0.0001)) ## x intercept of the line. using max to avoid divide b zero error
	y_icpt=rho/np.sin(max(theta,0.0001)) ## y intercept of the line

	if(theta>np.pi*80/180 and theta<np.pi*100/180): ## line is nearly horizontal

		if(rho<top_edge[0]):  ## rho for a possible top edge will be as small as possible
			top_edge=[rho,theta]
		if(rho>bottom_edge[0]):  ## rho for a possible bottom edge will be as large as possible
			bottom_edge=[rho,theta]

	elif(theta<np.pi*10/180 or theta>np.pi*170/180):   ## line is nearly vertical
		if(x_icpt>right_edge_x_icpt):  ## x intercept for right edge will be as large as possible
			right_edge=[rho,theta]
			right_edge_x_icpt=x_icpt
		elif(x_icpt<left_edge_x_icpt):  ## x intercept for left edge will be as small as possible
			left_edge=[rho,theta]
			left_edge_x_icpt=x_icpt

top_edge[1]=max(top_edge[1],0.0001)
bottom_edge[1]=max(bottom_edge[1],0.0001)
left_edge[1]=max(left_edge[1],0.0001)
right_edge[1]=max(right_edge[1],0.0001)
## finding the intersection points of the 4 edges
[pt_top_left,pt_top_right,pt_bottom_left,pt_bottom_right]= cal_intersection_points(top_edge,left_edge,bottom_edge,right_edge,width,height)
max_edge_length=int(find_longest_edge(pt_top_left,pt_top_right,pt_bottom_left,pt_bottom_right))

src=np.array([pt_top_left,pt_top_right,pt_bottom_left,pt_bottom_right],dtype='float32')
dst=np.array([[0,0],[max_edge_length-1,0],[0,max_edge_length-1],[max_edge_length-1,max_edge_length-1]],dtype='float32')
transformation_matrix=cv2.getPerspectiveTransform(src,dst)

## applying a perspective transform to the image
corrected_img=cv2.warpPerspective(img,transformation_matrix,(max_edge_length,max_edge_length))
#print(pt_top_left)
# cv2.circle(line_test,(int(pt_bottom_right[0]),int(pt_bottom_right[1])),5,255,-1)
# cv2.circle(line_test,(int(pt_bottom_left[0]),int(pt_bottom_left[1])),5,255,-1)
# cv2.circle(line_test,(int(pt_top_left[0]),int(pt_top_left[1])),5,255,-1)
# cv2.circle(line_test,(int(pt_top_right[0]),int(pt_top_right[1])),5,255,-1)

adap_thresh_img = cv2.adaptiveThreshold(corrected_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,1)
mypuzzle=cv2.bitwise_not(adap_thresh_img)


grid_length=int(np.ceil(max_edge_length/9))

c1=0
for i in range(9):
	c2=0
	for j in range(9):
		grid_width=min((j+1)*grid_length-c2 , max_edge_length-c2)
		grid_height=min((i+1)*grid_length-c1,max_edge_length-c1)

		mygrid=mypuzzle[c1:c1+grid_height,:][:,c2:c2+grid_width]
		c2=c2+grid_width	

		#resized_grid=cv2.resize(mygrid,(100,100))

		# M=cv2.moments(mygrid,True) ## true parameter will make all non-zero image values as 1 (binarization of image)
		# sum_of_white_pixels=M["m00"]
		# if(sum_of_white_pixels>=grid_width*grid_height/5): ## considering grids with a good amount of white pixels
		#retain_largest_blob(mygrid,grid_height,grid_width)
		cv2.imshow('grid',mygrid)
		#text=pytesseract.image_to_string(mygrid,config=config)

		cv2.waitKey(0)
	c1=c1+grid_height
#cv2.imshow('my image',mypuzzle)
#cv2.imwrite('Cornersdetected.jpg',line_test)
