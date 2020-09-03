clc;clear;close all

%% Getting Image
% offset=1;
% offset1=1;
% myFolder='E:\bin\malignant_final';
% filePattern=fullfile(myFolder, '*.pgm');
% pgmFiles=dir(filePattern);
% 
i=imread('N (71).pgm');
% for d=1:length(pgmFiles)
%     baseFileName=pgmFiles(d).name;
%     fullFileName=fullfile(myFolder, baseFileName);
% %     clc;clear;close all
%     fprintf(1,'Now reading %s' ,fullFileName);
% %     pause(5)
%     i=imread(fullFileName);



figure(1)
%i=flip(i,2)
imnoisy=i;
imshow(i);title('Original Photo')
[peaksnr,snr]=psnr(imnoisy,i);

% if image is rgb
try
    i=rgb2gray(i);
end
t=graythresh(i);
image=im2bw(i,173/255);
[x,y]=size(i);

rect=[1 1 y-101 x]
i=imcrop(i,rect)

%image = im2bw(i,1.64*t)
%image = im2bw(i,2.3*t)
%disp(i(1,1))
topRows=image(1:5,:);
botRows=image(end-4:end,:);
leftCols=image(:,1:5);
rightCols=image(:,end-4:end);
sumt=sum(sum(topRows));
sumb=sum(sum(botRows));
suml=sum(sum(leftCols));
sumr=sum(sum(rightCols));
if(sumt>=340)
    if(sumr>=1900)
        i=fliplr(i);
        image=fliplr(image);
    end
elseif(sumb>=500)
        i=flipud(i);
        image=flipud(image);
        if(sumr>=500)
            i=fliplr(i);
            image=fliplr(image);
        end
end
[x,y]=size(i);
rect=[476 1 y x]
i=imcrop(i,rect)
figure(100)
imshow(i)

%% Crop The Breast
%t=otsuthresh(imhist(i));
%z=im2bw(i,1.17*t);
z=im2bw(i,0.15);
figure(2)
imshow(z);title('Original B&W')
% fprintf(1,'Now reading %s' ,fullFileName);
info=regionprops(z);
a=cat(1,info.Area);
[m,l]=max(a);
X=info(l).Centroid;
bw2=bwselect(z,X(1),X(2),8);
i=immultiply(i,bw2);
figure(3)
imshow(i);
title('Getting the Breast and Muscle')

%% Deleting Black Ground

% We will delete the black corners
% So that we can select the muscle
% using bwselect
% convert to B&W first time
[x,y]=size(z);
tst1=zeros(x,y);

% detect empty rows
r1=[];
m=1;
for j=1:x
    if z(j,:)==tst1(j,:)
        r1(m)=j;
        m=m+1;
    end
end

% detect empty columns
r2=[];
m=1;
for j=1:y
    if z(:,j)==tst1(:,j)
        r2(m)=j;
        m=m+1;
    end
end

% Deleting
i(:,r2)=[];
i(r1,:)=[];

figure(4)
imshow(i);title('after deleting background');


%% Deleting the Muscle
[x,y]=size(i);
pixel_avg1=mean(i,'native')
%bw=watershed(i,8);
%i1=i(1:50,1:50);
%imean=mean(i1,'all');
%t=imean/256;
%figure(50)
%imshow(bw);
%mask=fspecial('average',[3 3]);
%SNR=1.9
%SNR=0.2;
%i=deconvwnr(i,mask,SNR);
%figure(6)
%imshow(i)
%title('Weiner Filter')

% Clahe Filter
%i=adapthisteq(i);
%figure(7)
%imshow(i)
%title('Clahe Filter')

%i=imclearborder(i,4);
%i=imadjust(i);
%seg=imsegkmeans(i,3);
% t=multithresh(i,3);
% image=imquantize(i,t);
% figure(25)
% imshow(image,[])
if(pixel_avg1>115)
    t=otsuthresh(imhist(i));
    image=im2bw(i,0.1*t)
    figure(25)
    imshow(image)
else
    image=im2bw(i,173/255);
    figure(26)
    imshow(image)
end

    pixel_avg=mean(image,'native')
    %L=bwlabel(image,4);
% for a=1:x
%     for b=1:y
%         if(seg(a,b)==1)
%             image(a,b)=1;
%         else
%             image(a,b)=0;
%         end
%     end
% end
                
%image=im2bw(i,t);
%[peaksnr,snr]=psnr(ref,i)
%image=imbinarize(i,'global');
if(pixel_avg>0.2)    %keep 0.2
    figure(23)
    imshow(image)
    i1=imclearborder(image,8);
    figure(24)
    imshow(i1)
    image = xor(i1,image);
    % Blacken the pectoral region in the original image.
    %i(pectoralOnly) = 0;
%else
end  

%image=i;
%image=bwselect(i,300,10,8);
  %bw1=grayconnected(i,10,64);  %bw1=grayconnected(i,10,900);
%bw2=bw1+(grayconnected(i,10,350));
%bw3=bw2+grayconnected(i,10,140);
%if i(10,10)>=50
%    c=44;
%    r=3;
%else 
%    %i=flip(i,2);
%    r=3;
%    c=size(i,2)-15;
%    disp('this is executing')
%end
%bw1=imfill(bw);
 %figure(5);
 %imshow(bw1);
z2=bwselect(image,25,10,8);  % z2=bwselect(bw1,800,10,8);
%z2=z2+bwselect(image,140,10,8);
figure(6)
imshow(z2);

% fprintf(1,'Now reading %s' ,fullFileName);

z2=activecontour(i,z2,250);
% %z2=imfill(z2,8,'holes');
% figure(7)
imshow(z2);
mask1=~z2;
i=immultiply(i,mask1);
%end
figure(8)
imshow(i)

% if(pixel_avg2<0.75)
%     mask1=~z2;
%     i=immultiply(i,mask1);
%     figure(8)
%     imshow(i)
% else
%     mask2=z2;
%     i=immultiply(i,mask2);
%     figure(8)
%     imshow(i);
% end
title('Getting only the Breast')



%% Filtering image
%Weiner Filter
% We will create average mask [3 3]
% with SNR = 0.2
mask=fspecial('average',[3 3]);
SNR=0.2;
i=deconvwnr(i,mask,SNR);
figure(9)
imshow(i)
title('Weiner Filter')

%Non Localised mean filter
% [filtered_image,estDos]=imnlmfilt(i);
% [image_filt2,estDos2]=imnlmfilt(filtered_image,'SearchWindowSize',25);
% title('applying non localized mean filter')

% Clahe Filter
i=adapthisteq(i);
figure(10)
imshow(i)
title('Weiner Filter+Clahe Filter+Non localized mean filte')


segmented=segmentImage(i)


% %% Segmentation
% 
% z=im2bw(b,0.57); %varies btw 0.45 to 0.57
% info=regionprops(z);
% a=cat(1,info.Area);
% [m,l]=max(a);
% X=info(l).Centroid;
% bw2=bwselect(z,X(1),X(2),8);
% z=immultiply(b,bw2);
% z1=medfilt2(z,[2,2]);
% figure(11)
% imshow(z1);
% title('Segmented image using method 1')
% 
% %b=imread('mdb029ll.pgm');
% %seg=watershed(b)
% %mask1=false(size(i))
% %mask1(75:end-100,200:end-400) = true;
% %bw = activecontour(i, mask1, 200);
% %figure
% %imshow(seg)
% 
% %% segmentation
% %image=im2double(i)
% %[x,y]=size(image)
% %temp=0;
% %max=0;
% %for i = 1:1:x
% %    for j = 1:1:y
% %        temp1=image(i,j);
% %        if(temp1>max)
% %            max=temp1;
% %        end
% %    end
% %end
% %disp(max);
% %[x,y]=size(image);
% %t=135
% %for i = 1:1:x
% %    for j = 1:1:y
% %        if(image(i,j)<t)
% %            image(i,j)=0;
% %        end
% %   end
% %end
% %figure(12)
% %title('segmented image');
% %imshow(image);
% %bw=im2bw(image,max/1.80); 
% %final=medfilt2(bw,[2,2]);
% %figure(12)
% %title('Segmented image using method 2')
% %imshow(final)
% 
% for i=1:1:x
%     for j=1:1:y
%         if(image(i,j)>=160)
%             b(i,j)=255;
%         elseif(image(i,j)<160)
%             b(i,j)=0;
%         end
%     end
% end
% figure(13)
% title('segmentation method 3');
% imshow(b)
    figure(11)
   
    imshow(segmented) 
    title('SEGMENTATION')

%% Feature Extraction
%clc;
%clear all;
%close all;
%i=imread('mb5.pgm');
    ROI=segmented;
    seg_img=ROI
    
%     img = rgb2gray(ROI);
%         figure, imshow(img); title('Grayscale Image');
%         %figure, imshow(img); title('Gray Scale Image');
% 
%         % Create the Gray Level Cooccurance Matrices (GLCMs)
        glcms = graycomatrix(seg_img);

        % Derive Statistics from GLCM
        stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

        Contrast = stats.Contrast;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(seg_img);
        Standard_Deviation = std2(seg_img);
        Entropy = entropy(seg_img);
%         RMS = mean2(rms(seg_img));
        %Skewness = skewness(img)
        Variance = mean2(var(double(seg_img)));
        a = sum(double(seg_img(:)));
        Smoothness = 1-(1/(1+a));
        % Inverse Difference Movement
        m = size(seg_img,1);
        n = size(seg_img,2);
        in_diff = 0;
        for i = 1:m
            for j = 1:n
                temp = seg_img(i,j)./(1+(i-j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);

        feat_disease = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, Variance, Smoothness, IDM]
        
        benign_malignant_array=feat_disease;
% % % %     GLCM2=graycomatrix(bw,'Offset',[2 0;0 2]);
% % % % 
% ROI=segmented;
% % % %     GLCM2=graycomatrix(bw,'Offset',[2 0;0 2]);
% % % % 
% % % %     stats=GLCM_Features1(GLCM2,0)

% Use the Extracted ROI named as "ROI" in the uploaded Images
    % Use the following angles "offset":(0,45,90,135)
    GLCM = graycomatrix(ROI,'Offset',[0 1;-1 1;-1 0;-1 -1]);
    % Calculate the four built-in MATLAB features
    stats=graycoprops(GLCM,{'Contrast','Homogeneity','Correlation','Energy'});
    contrast=(stats.Contrast);  
    en=(stats.Energy);
    co=(stats.Correlation);
    hom=(stats.Homogeneity);
    % Calculate mean and standard deviation
    m=mean(mean(ROI));    
    s=std2((ROI));
    % The first feature vector 
    f1=[m s hom co en contrast];
    % Calculate the texture Features 
    % Picture used is" segmented ROI" in the uploaded images
    % This picture is the segmentation of " ROI " picture used at the 
    % beginning of the code
    % Features to be extracted " Area, Perimeter, compactness, smoothness, entropy,..."
    J2=ROI;
    I=J2;  
    J2 = uint8(255 * mat2gray(J2));
    % Detecting the edges of the ROI
    J3=edge(J2,'log');    
    bw=bwareaopen(J2,150);                          
    bwp=edge(bw,'sobel');
    geometric.area=sum(sum(bw));
    geometric.peri=sum(sum(bwp));
    geometric.compact=geometric.peri^2/geometric.area;
    count=1;
    roih=0;
    for i=1:size(bw,1);
          for j=1:size(bw,2);
              if bw(i,j)
                  roih(count,1)=double(ROI(i,j)); 
                  count=count+1;
              end
          end
      end
    nh=roih/geometric.area;
    texture.mean=mean(roih);
    texture.globalmean=texture.mean/mean(mean(double(I)));
    texture.std=std(roih);
    texture.smoothness=1/(1+texture.std^2);
    texture.uniformity=sum(nh.^2);
    texture.entropy=sum(nh.*log10(nh));    
    texture.skewness=skewness(roih);
    texture.correlation=sum(nh'*roih)-texture.mean/texture.std;
    yy=[texture.mean,texture.globalmean,texture.std,texture.smoothness,texture.uniformity,...
    texture.entropy,texture.skewness,texture.correlation];
%     y1=zeros()
%     x1=zeros()
    %y1=[0;yy];
    xy=[geometric.area,geometric.peri,geometric.compact];
    %x1=[0;xy];
    f2=[xy yy];
    Totalfeaures=[f1,f2];
    StatsTable=struct2table(stats);

% writetable(struct2table(stats), 'a1.xlsx', 'WriteVariableNames', false)
    normal_cancerous_array=table2array(StatsTable)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Normal_Cancerous_dataset = xlsread('G:\Matlab\R2019a\bin\All\Normal_Cancerous.xlsx');
    Cancerous_Benign_Malignant = xlsread('E:\bin\All\Cancerous_Benign_Malignant.xlsx');
%save('data','dataset')
%a=load ('data')
% cancertype=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
Normal_Cancerous_Labels=[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
Cancerous_BenignMalignant_Labels = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2];
%cancertype=[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2];
% result= multisvm(Normal_Cancerous_dataset,cancertype,StatsArray)
svmModel_NormalCancerous = svmtrain(Normal_Cancerous_dataset,Normal_Cancerous_Labels);  
result_1 = svmclassify(svmModel_NormalCancerous,normal_cancerous_array);
% xlswrite('rl.xlsx',result_1,sprintf('A%d',offset));        
% offset=offset+1;

if result_1 == 1
    svmModel_CancerousBenignMalignant = svmtrain(Cancerous_Benign_Malignant,Cancerous_BenignMalignant_Labels);  
    result_final = svmclassify(svmModel_CancerousBenignMalignant,benign_malignant_array);
%     xlswrite('r_f.xlsx',result_final,sprintf('A%d',offset1));        
%     offset1=offset1+1;

    if(result_final == 1)
        sprintf('Given image is Benign')
    else
        sprintf('Given image is Malignant')
    end
else
    sprintf('Given image is normal')
end

% print(classes)
% out=predict(Model,StatsArray)

    
%     net=load('trainedModel87')
%     YPred=classify(net,StatsArray)
%     required=categories(YPred);

%     StatsArray'
%     for(i=1:11)
%         features(i)=(StatsArray(2*i)+StatsArray(2*i-1))/2;
%     end
% % cellRef = sprintf('A%d', j)

%     xlswrite('1.xlsx',StatsArray,1,sprintf('A%d',offset));
%     offset=offset+1;

% csvwrite('test.csv',features,1,0)

% data=csvread('G:\Matlab\R2019a\bin\Breast-Cancer-Neural-Networks-master\BreastCancer_Neural\cancer_data.csv');

% writematrix(features,'test.csv',1,'A2')

% writematrix(features,'test.csv')
% writetable(struct2table(StatsArray), 'a1.xlsx', 'WriteVariableNames', false
