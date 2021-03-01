def generate_fasthead_train_data_v3(loc_data):
    pos_num=0
    loc_data=loc_data[0]
    #out_clsv=np.zeros((samplenum,21))
    #out_regv=np.zeros((samplenum,21,4))
    #cls_no=np.full((samplenum),-1)
    #reg_no=np.full((samplenum,21,4),-1)
    for i in range(len(loc_data)):
        ioulist=[]
        lefttop_h=loc_data[i][0]
        lefttop_w=loc_data[i][1]
        rightbottom_h=loc_data[i][2]
        rightbottom_w=loc_data[i][3]
        gt_cls=loc_data[i][4]
        gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        #print(gt_box)
        for h in range(14):
            for w in range(14):
                for c in range(9):
                    boxtmp=generate_box(h,w,c)
                    #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        pos_num=pos_num+1
    pos_num+=len(loc_data)
    #将负样本数量设置为正样本总数的1/7
    if pos_num>0:
        samplenum=pos_num+int(pos_num/7)
        
    else:
        print('no pos')
        samplenum=1
    if samplenum-pos_num==0:
        gailv=np.random.random()
        if gailv<3*(pos_num/21):
            samplenum+=1
            
    #print(pos_num,samplenum)
    out_clsv=np.zeros((samplenum,21))
    out_regv=np.zeros((samplenum,21,4))
    cls_no=np.full((samplenum),-1)
    reg_no=np.full((samplenum,21,4),-1)
    
    boxlist=[]
    ser=0
    neglist=[]
    for i in range(len(loc_data)):
        lefttop_h=loc_data[i][0]
        lefttop_w=loc_data[i][1]
        rightbottom_h=loc_data[i][2]
        rightbottom_w=loc_data[i][3]
        gt_cls=loc_data[i][4]
        gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        out_clsv[ser][gt_cls]=1
        out_regv[ser][gt_cls][0]=0
        out_regv[ser][gt_cls][1]=0
        out_regv[ser][gt_cls][2]=0
        out_regv[ser][gt_cls][3]=0
        cls_no[ser]=gt_cls
        reg_no[ser][gt_cls][0]=0
        reg_no[ser][gt_cls][1]=0
        reg_no[ser][gt_cls][2]=0
        reg_no[ser][gt_cls][3]=0
        
        boxlist.append(gt_box)
        ser=ser+1
    for h in range(14):
        for w in range(14):
            for c in range(9):
                boxtmp=generate_box(h,w,c)
                ifneg=0
                #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                for i in range(len(loc_data)):
                    #print(loc_data[i])
                    ioulist=[]
                    lefttop_h=loc_data[i][0]
                    lefttop_w=loc_data[i][1]
                    rightbottom_h=loc_data[i][2]
                    rightbottom_w=loc_data[i][3]
                    gt_cls=loc_data[i][4]
                    gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        #print('ok',i)
                        anchor_lefttop_h=boxtmp[0]
                        anchor_lefttop_w=boxtmp[1]
                        anchor_rightbottom_h=boxtmp[2]
                        anchor_rightbottom_w=boxtmp[3]
                            
                        ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                        xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                        ha=anchor_rightbottom_h-anchor_lefttop_h
                        wa=anchor_rightbottom_w-anchor_lefttop_w
                            
                        if ha==0:
                            ha=0.01
                        if wa==0:
                            wa=0.01
                        
                        gt_lefttop_h=lefttop_h
                        gt_lefttop_w=lefttop_w
                        gt_rightbottom_h=rightbottom_h
                        gt_rightbottom_w=rightbottom_w
                            
                        gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                        gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                        gt_h=gt_rightbottom_h-gt_lefttop_h
                        gt_w=gt_rightbottom_w-gt_lefttop_w
                        ty=(gt_y-ya)/ha
                        tx=(gt_x-xa)/wa
                        tw=np.log(gt_w/wa)
                        th=np.log(gt_h/ha)
                        #print(ser,gt_cls)
                        gt_cls=int(gt_cls)
                        out_clsv[ser][gt_cls]=1
                        out_regv[ser][gt_cls][0]=ty
                        out_regv[ser][gt_cls][1]=tx
                        out_regv[ser][gt_cls][2]=th
                        out_regv[ser][gt_cls][3]=tw
                        cls_no[ser]=gt_cls
                        reg_no[ser][gt_cls][0]=0
                        reg_no[ser][gt_cls][1]=0
                        reg_no[ser][gt_cls][2]=0
                        reg_no[ser][gt_cls][3]=0
                        
                        boxlist.append(boxtmp)
                        ser=ser+1
                    if iou<=0.1:
                        ifneg=ifneg+1
                    if ifneg==len(loc_data):
                        neglist.append(boxtmp)
                        #out_clsv[ser][gt_cls]=1
                        #boxlist.append(boxtmp)
                        #cls_no[ser]=0
                        #reg_no[ser][gt_cls][0]=0
                        #reg_no[ser][gt_cls][1]=0
                        #reg_no[ser][gt_cls][2]=0
                        #reg_no[ser][gt_cls][3]=0
                        #ser=ser+1
    neg_num=0
    
    while neg_num<samplenum-pos_num:
        #print(ser)
        sjs=np.random.randint(0,len(neglist))
        out_clsv[ser][20]=1
        boxlist.append(neglist[sjs])
        cls_no[ser]=20
        #reg_no[ser][gt_cls][0]=0
        #reg_no[ser][gt_cls][1]=0
        #reg_no[ser][gt_cls][2]=0
        #reg_no[ser][gt_cls][3]=0
        ser=ser+1
        neg_num=neg_num+1
    out_regv=np.reshape(out_regv,[-1,84])
    reg_no=np.reshape(reg_no,[-1,84])
    boxlist=split_box(boxlist)
    return boxlist,out_clsv,out_regv,cls_no,reg_no