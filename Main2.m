clc
clear
close all
%% Reading Images
cd 'C:\Users\mohammad\Desktop\project\images' % go to images folder to get input images
Img_list=dir; % take a list from the current diroctory
cd .. % come back to before dir

cd 'C:\Users\mohammad\Desktop\project\masks' % go to masks folder to get mask images
Img_list2=dir; % take a list from the current diroctory
cd .. % come back to before dir
    
cd 'C:\Users\mohammad\Desktop\project\annotations' % go to annotations folder to get ground truth images
Img_list3=dir; % take a list from the current diroctory
cd .. % come back to before dir

%Features=zeros(12,1997);
 counter=1;
%ClassVect=zeros(2,1997);

% load TN.mat
% TN=TN';
%  counter=1;
%  Features=zeros(241,1);
 
 
%% Reading Images 
for i=1:20

fullname = Img_list(i+2).name; % because the first and second elemnts of Img_list are cd. and cd.. folders we begine with the third element of Img_list
cd 'C:\Users\mohammad\Desktop\project\images' % go to images folder to get input images
MainImg=imread(fullname); % read the image
% MainImg=rgb2gray(MainImg); % convort rgb imge to gray one
MainImg=imresize(MainImg,[810 810]); % adjust the image size
[r, c]=size(MainImg);


fullname2=Img_list2(i+2).name; % because the first and second elemnts of Img_list2 are cd. and cd.. folders we begine with the third element of Img_list2
cd 'C:\Users\mohammad\Desktop\project\masks' % go to masks folder to get input images
MaskImg=imread(fullname2); % read the mask image


fullname3=Img_list3((2*i)+1).name;  % because the first and second elemnts of Img_list3 are cd. and cd.. folders we begine with the third element of Img_list3
cd 'C:\Users\mohammad\Desktop\project\annotations' % go to annotations folder to get input images
LabeledImg=imread(fullname3); % read the ground truth image
LabeledImg=imresize(LabeledImg,[810 810]); % adjust the image size
cd .. % come back to before dir

%% Removing The Background Of Image
MaskImg=imresize(MaskImg,[810 810]);
for i1=1:r
    for j1=1:c/3
        if MaskImg(i1,j1,:)==1
            
            MainImg(i1,j1,:)=0;
        end
    end
end


%% Labeling 

for i3=1:810
    for j3=1:810

        if LabeledImg(i3,j3,1)==255 % for red pixles
            LabeledImg(i3,j3,1)=1;
        elseif LabeledImg(i3,j3,2)==255 % for green pixles
           LabeledImg(i3,j3,2)=2;
       
       end
    end
end

n=80; % Size of each block
BlockedImg=zeros(n); % initialization
n2=15;
%% Blocking The Image Into n*n block

for i2=1:n:810
     for j2=1:n:810
          if i2+(n-1)<=810 && j2+(n-1)<=810 % not to exceed than range of metrix
          
              BlockedImg2=MainImg(i2:i2+(n-1),j2:j2+(n-1),:);
              BlockedImg=rgb2gray(BlockedImg2);
              BlockedImg=im2double(BlockedImg);

%% Determining class of each Block

 if nnz(BlockedImg)>=200 % to prevent to enter whole black blocks
[rc1,cc1]=find(LabeledImg(i2:i2+(n-1),j2:j2+(n-1),1)==1); % find the pixels with label 1
[rc2,cc2]=find(LabeledImg(i2:i2+(n-1),j2:j2+(n-1),2)==2); % find the pixels with label 2

% figure();imshow(BlockedImg2)
% figure();imshow(BlockedImg)

if nnz(rc1)*nnz(cc1)>nnz(rc2)*nnz(cc2) % select the current block include most crop or weed pixles
Class=1;
else 
Class=2;
end
     
if Class==1    
ClassVect(:,counter)=[1 0];
elseif Class==2
ClassVect(:,counter)=[0 1];
end
%% Extracing Features
%Chanels
HSV=rgb2hsv(BlockedImg2);
[h ,s, v] = imsplit(HSV);

R=BlockedImg2(:,:,1);
G=BlockedImg2(:,:,2);
B=BlockedImg2(:,:,3);

% Statistical Features from chanel gray
[r,c]=size(BlockedImg);
b=reshape(BlockedImg,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero);
SkewnessGray=t(4);% t(4)= skewness of histogram

VGray=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SGray=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KGray=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% Feature 1
min_image = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_image = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_image = max(reshape(BlockedImg,[r*c,1]));% max of the blocke

FeatuesGray=[SkewnessGray,VGray,SGray,KGray,min_image,mean_image,max_image];

% Statistical Features from chanel R
[r,c]=size(R);
b=reshape(R,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessR=t(4);% t(4)= skewness of histogram

VR=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SR=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KR=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));


% Feature 1
min_imageR = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageR = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageR = max(reshape(BlockedImg,[r*c,1]));% max of the block

[Gmag, Gdir] = imgradient(R,'prewitt');

[r,c]=size(Gmag);
b=reshape(Gmag,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessGGra=t(4);% t(4)= skewness of histogram

VGGra=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SGGra=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KGGra=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageGGra= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageGGra = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageGGra = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

sigma = 0.4;
alpha = 0.5;
GLaplacian = locallapfilt(R, sigma, alpha);
% imshow(BLaplacian);

[r,c]=size(GLaplacian);
b=reshape(GLaplacian,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessGLap=t(4);% t(4)= skewness of histogram

VGLap=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SGLap=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KGLap=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageGLap= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageGLap= mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageGLap = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

FeaturesGLap=[SkewnessGLap,VGLap,SGLap,KGLap,min_imageGLap,mean_imageGLap,max_imageGLap];
FeaturesGGra=[SkewnessGGra,VGGra,SGGra,KGGra,min_imageGGra,mean_imageGGra,max_imageGGra];
FeaturesR=[SkewnessR,VR,SR,KR,min_imageR ,mean_imageR,max_imageR];
AllFeaturesR=[FeaturesGGra,FeaturesR,FeaturesGLap];

% Statistical Features from chanel G
[r,c]=size(G);
b=reshape(G,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessG=t(4);% t(4)= skewness of histogram

VG=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SG=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KG=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageG= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageG = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageG = max(reshape(BlockedImg,[r*c,1]));% max of the block

[Gmag, Gdir] = imgradient(G,'prewitt');

[r,c]=size(Gmag);
b=reshape(Gmag,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessGGra=t(4);% t(4)= skewness of histogram

VGGra=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SGGra=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KGGra=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageGGra= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageGGra = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageGGra = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

sigma = 0.4;
alpha = 0.5;
GLaplacian = locallapfilt(G, sigma, alpha);
% imshow(BLaplacian);

[r,c]=size(GLaplacian);
b=reshape(GLaplacian,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessGLap=t(4);% t(4)= skewness of histogram

VGLap=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SGLap=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KGLap=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageGLap= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageGLap= mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageGLap = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

FeaturesGLap=[SkewnessGLap,VGLap,SGLap,KGLap,min_imageGLap,mean_imageGLap,max_imageGLap];
FeaturesGGra=[SkewnessGGra,VGGra,SGGra,KGGra,min_imageGGra,mean_imageGGra,max_imageGGra];
FeaturesG=[SkewnessG,VG,SG,KG,min_imageG,mean_imageG,max_imageG];
AllFeaturesG=[FeaturesGGra,FeaturesG,FeaturesGLap];

LBPGLap = extractLBPFeatures(GLaplacian,'Upright',false);
LBPGGra = extractLBPFeatures(Gmag,'Upright',false);

% Statistical Features from chanel B
[r,c]=size(B);
b=reshape(B,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessB=t(4);% t(4)= skewness of histogram

VB=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SB=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KB=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% Feature 1
min_imageB = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageB = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageB = max(reshape(BlockedImg,[r*c,1]));% max of the block

sigma = 0.4;
alpha = 0.5;
BLaplacian = locallapfilt(B, sigma, alpha);
% imshow(BLaplacian);

[r,c]=size(BLaplacian);
b=reshape(BLaplacian,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessBLap=t(4);% t(4)= skewness of histogram

VBLap=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SBLap=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KBLap=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageBLap= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageBLap= mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageBLap = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

FeaturesBLap=[SkewnessBLap,VBLap,SBLap,KBLap,min_imageBLap,mean_imageBLap,max_imageBLap];
FeaturesB=[SkewnessB,VB,SB,KB,min_imageB,mean_imageB,max_imageB];
AllFeaturesB=[FeaturesBLap,FeaturesB];

% Statistical Features from chanel H
[r,c]=size(h);
b=reshape(h,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessH=t(4);% t(4)= skewness of histogram

VH=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SH=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KH=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% Feature 1
min_imageH = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageH = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageH = max(reshape(BlockedImg,[r*c,1]));% max of the block
% h= round(h);
% sigma = 0.4;
% alpha = 0.5;
% HLaplacian = locallapfilt(h, sigma, alpha);
% % imshow(BLaplacian);
% 
% [r,c]=size(HLaplacian);
% b=reshape(HLaplacian,[r*c,1]); % change BlockedImg into a vector
% indices=find(b);  % find the pixles which there are plants 
% BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
% NumNonzero=nnz(b); % find the number of pixles which there are plants
% 
% t = statxture (BlockedImgNonZero); 
% BlockedImgNonZero=im2double(BlockedImgNonZero);
% SkewnessHLap=t(4);% t(4)= skewness of histogram
% 
% VHLap=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
% SHLap=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
% KHLap=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% % Feature 1
% min_imageHLap= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% % Feature 2
% mean_imageHLap= mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% % Feature 3
% max_imageHLap = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block
% 
% FeaturesHLap=[SkewnessHLap,VHLap,SHLap,KHLap,min_imageHLap,mean_imageHLap,max_imageHLap];
FeaturesH=[SkewnessH,VH,SH,KH,min_imageH,mean_imageH,max_imageH];
% AllFeaturesH=[FeaturesHLap,FeaturesH];

% Statistical Features from chanel S
[r,c]=size(s);
b=reshape(s,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessS=t(4);% t(4)= skewness of histogram

VS=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SS=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KS=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% Feature 1
min_imagesS = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imagesS = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imagesS = max(reshape(BlockedImg,[r*c,1]));% max of the block

[Gmag, Gdir] = imgradient(s,'prewitt');

[r,c]=size(Gmag);
b=reshape(Gmag,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessSGra=t(4);% t(4)= skewness of histogram

VVGra=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SVGra=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KVGra=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageSGra= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageSGra = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageVGra = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

FeaturesVGra=[SkewnessSGra,VVGra,SVGra,KVGra,min_imageSGra,mean_imageSGra,max_imageVGra];
FeaturesS=[SkewnessS,VS,SS,KS,min_imagesS,mean_imagesS,max_imagesS];
AllFeaturesS=[FeaturesVGra,FeaturesS];

% Statistical Features from chanel V
[r,c]=size(v);
b=reshape(v,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessV=t(4);% t(4)= skewness of histogram

VV=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SV=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KV=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% Feature 1
min_imageV = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageV = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageV = max(reshape(BlockedImg,[r*c,1]));% max of the block

[Gmag, Gdir] = imgradient(v,'prewitt');

[r,c]=size(Gmag);
b=reshape(Gmag,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

t = statxture (BlockedImgNonZero); 
BlockedImgNonZero=im2double(BlockedImgNonZero);
SkewnessVGra=t(4);% t(4)= skewness of histogram

VVGra=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
SVGra=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
KVGra=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));
% Feature 1
min_imageVGra= min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageVGra = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageVGra = max(reshape(BlockedImgNonZero,[NumNonzero,1]));% max of the block

FeaturesVGra=[SkewnessVGra,VVGra,SVGra,KVGra,min_imageVGra,mean_imageVGra,max_imageVGra];
FeaturesV=[SkewnessV,VV,SV,KV,min_imageV,mean_imageV,max_imageV];
AllFeaturesV=[FeaturesVGra,FeaturesV];

LBPVGra = extractLBPFeatures(Gmag,'Upright',false);

% Shape features

Im_gray=BlockedImg; % copy BlockedImg
% figure();imshow(Im_gray)
Im_med=medfilt2(Im_gray); % median filtering 
% figure();imshow(Im_gray)
Im_bw=im2bw(Im_med,graythresh(Im_med)); % thresholding
% if sum(sum(Im_bw))/numel(Im_bw)>0.5
%     Im_bw=~Im_bw;
% end
% figure();imshow(Im_bw)
prop=regionprops(Im_bw);
ar=cat(1,prop.Area);
[Ar,l]=max(ar); 
 %First Property Of Regionprops

Im_bw=bwareaopen(Im_bw,abs(Ar-50)); % remove all objects in the image containing fewer than (Ar - 50) pixels

e=bwperim(Im_bw); % find perimiter of plant in the Im_bw
premiter=nnz(e);  % cal perimiter
 %Second
conv_hull=bwconvhull(Im_bw); % compute the covex hull of all objects in the Im_bw
convex_hull=sum(sum(conv_hull)); % cal covex hull
 %Third
major=regionprops(Im_bw,'MajorAxisLength'); % cal MajorAxisLength
major=major.MajorAxisLength;
 %Fourth
minor=regionprops(Im_bw,'MinorAxisLength'); % cal MinorAxisLength
minor=minor.MinorAxisLength;
 %Fifth
major_minor=major/minor; % cal MajorAxisLength/MinorAxisLength
%Last

compactness=Ar/(premiter^2);% compactness (area / perimeter2)
solidity=Ar/convex_hull;% solidity (area / area of convex hull)
convexity=premiter/nnz(bwperim(conv_hull));% convexity (perimeter / perimeter of convex hull)

FeaturesShapes=[Ar,premiter,convex_hull,major,minor,major_minor,compactness,solidity,convexity]; % ghathering Regionprops features together

LBPGray = extractLBPFeatures(BlockedImg,'Upright',false);
AllLBPFeatures=[LBPGray,LBPVGra,LBPGGra,LBPGLap];
AllFeatures(:,counter)=[AllLBPFeatures,FeatuesGray,AllFeaturesR,AllFeaturesG,AllFeaturesB,FeaturesH,AllFeaturesS,AllFeaturesV,FeaturesShapes]'; % ghathering all features together

counter=counter+1; % plus one the counter
 
end
end
end
end    
end

save AllFeatures