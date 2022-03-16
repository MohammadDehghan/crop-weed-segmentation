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

[r,c]=size(BlockedImg);
b=reshape(BlockedImg,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants



% Feature 1
min_image = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_image = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_image = max(reshape(BlockedImg,[r*c,1]));% max of the block

% Features3=feature_vec(BlockedImg); %calculate the seven invariant moments

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
% t = statxture (BlockedImgNonZero); % t(4)= skewness of histogram

t = statxture (BlockedImg(Im_bw)); % t(4)= skewness of histogram
% % Feature 1
% % min_image = min(BlockedImg(Im_bw));% min of the block
% % Feature 2
% % mean_image = mean(BlockedImg(Im_bw));% mean of the block
% % Feature 3
% % max_image = max(BlockedImg(Im_bw));% max of the block
BlockedImgGlcm=BlockedImg;
for i=1:80
    for j=1:80
        if BlockedImgGlcm(i,j)==0
            BlockedImgGlcm(i,j)=NaN;
        end
    end
end
 warning('off', 'Images:graycomatrix:scaledImageContainsNan')
glcm = graycomatrix(BlockedImgGlcm,'Offset',[2 0]); % calculate co-occurence matrix for graycoprops
stats = graycoprops(glcm);
Contrast=stats.Contrast;
Correlation=stats.Correlation;
Energy=stats.Energy;
Homogeneity=stats.Homogeneity;

%   % zernike moment n=order and m=repition
%     [mom, amplitude, angle] = Zernikmoment(Im_bw,4,0);      % Call Zernikemoment fuction n=4, m=0
%     arr(1)=real(mom);
%     arr(2)=imag(mom);
%     arr(3)=amplitude;
%     clear mom amplitude angle
%     [mom, amplitude, angle] = Zernikmoment(Im_bw,4,2);      % Call Zernikemoment fuction n=4, m=2
%     arr(4)=real(mom);
%     arr(5)=imag(mom);
%     arr(6)=amplitude;
%     clear mom amplitude angle
%     [mom, amplitude, angle] = Zernikmoment(Im_bw,4,4);      % Call Zernikemoment fuction n=4, m=4
%     arr(7)=real(mom);
%     arr(8)=imag(mom);
%     arr(9)=amplitude;
%     clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,3,1);      % Call Zernikemoment fuction n=3, m=1
%     arr(10)=real(mom);
%     arr(11)=imag(mom);
%     arr(12)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,3,3);      % Call Zernikemoment fuction n=3, m=3
%     arr(13)=real(mom);
%     arr(14)=imag(mom);
%     arr(15)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,0,0);      % Call Zernikemoment fuction n=0, m=0
%     arr(16)=real(mom);
%     arr(17)=imag(mom);
%     arr(18)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,1,1);      % Call Zernikemoment fuction n=1, m=1
%         arr(19)=real(mom);
%     arr(20)=imag(mom);
%     arr(21)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,2,0);      % Call Zernikemoment fuction n=2, m=0
%         arr(22)=real(mom);
%     arr(23)=imag(mom);
%     arr(24)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,2,2);      % Call Zernikemoment fuction n=2, m=2
%         arr(25)=real(mom);
%     arr(26)=imag(mom);
%     arr(27)=amplitude;
%    clear mom amplitude angle

LBP = extractLBPFeatures(BlockedImg,'Upright',false);
% [featureVector,hogVisualization] = extractHOGFeatures(BlockedImg);
HOG(:,counter)=extractHOGFeatures(BlockedImg);

% [M]=feature_vec(BlockedImg);

HSV=rgb2hsv(BlockedImg2);
[h ,s, v] = imsplit(HSV);

[r,c]=size(h);
b=reshape(h,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants

% Feature 1
min_imageh = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imageh = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imageh = max(reshape(BlockedImg,[r*c,1]));% max of the block

[r,c]=size(s);
b=reshape(s,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants



% Feature 1
min_imagess = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imagess = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imagess = max(reshape(BlockedImg,[r*c,1]));% max of the block

[r,c]=size(v);
b=reshape(v,[r*c,1]); % change BlockedImg into a vector
indices=find(b);  % find the pixles which there are plants 
BlockedImgNonZero=b(indices(1:end)); % find the intensity of pixles which there are plants 
NumNonzero=nnz(b); % find the number of pixles which there are plants



% Feature 1
min_imagev = min(reshape(BlockedImgNonZero,[NumNonzero,1]));% min of the block
% Feature 2
mean_imagev = mean(reshape(BlockedImgNonZero,[NumNonzero,1]));% mean of the block
% Feature 3
max_imagev = max(reshape(BlockedImg,[r*c,1]));% max of the block



%  figure;
% imshow(BlockedImg); 
% hold on;
% plot(hogVisualization);
% 
 v2=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
 s2=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
k=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% load TN.mat
% TN=TN';
%  counter=1;
%  Features=zeros(91,1);
% indicesFeaturesSelected=[5,172,87,27,185,194,1,62,196,97,137,120,115,175,169,19,107,21,151,52,104,61,159,140,160,132,82,191,39,176]
% TN=TN(indicesFeaturesSelected(1:end),counter);
Features1=[max_image,min_image,mean_image,max_imagess,min_imagess,mean_imagess,max_imagev,min_imagev,mean_imagev,k,LBP,Contrast,Correlation,Energy,Homogeneity]; % ghathering max,min,mean features together
Features2=[Ar,premiter,convex_hull,major,minor,major_minor,compactness,solidity,convexity]; % ghathering Regionprops features together


MainFeatures(:,counter)=[Features1,Features2,t(4)]'; % ghathering all features together


% indicesFeaturesSelected=[23 15 17 19 24 25 10 6 7 12 9 29 20 5 8 18 4 11 21 30];
% FeaturesSelected(:,counter)= Features(indicesFeaturesSelected(1:end),counter);

% indicesFeaturesSelected=[6 27 40 37 34 8 17 3 19 15 14 22 12 10 9 29 25 31 20 28 18 24 11 4 30 41 1 35 21 13];
% FeaturesSelected(:,counter)= Features(indicesFeaturesSelected(1:end),counter);

counter=counter+1; % plus one the counter
 
end
end
end
end    
end
% save MainFeatures;
% MainFeaturesSelectedI=[2,3,4,6,11,22,26,27,28,29,31,32,33];
% MainFeaturesSelected=MainFeatures(MainFeaturesSelectedI(1:end),:);
results=CreateAndTrainANN(MainFeatures,ClassVect);
%% Performing PCA On HOG Features
% X=HOG;
% [Q lambda]=PerformPCA(X);
% TN=X'*Q(:,1:500);
% TN=TN';
% s=[159,78,25,328,387,150,320,73,181,449,168,149,358,259,137,249,63,441,309,459,438,10,362,446,444,37,398,295,395,479,319,430,23,308,176,421,90,71,427,108,227,224,222,351,167,270,97,187,238,173,59,341,451,4,360,189,483,464,491,276,254,161,32,293,477,481,3,145,359,114,412,288,439,89,499,490,84,347,332,185];
% selectedTn=TN(s(1:end),:);
% Features=[Features;selectedTn];
% save Features 
% % save TN
% save ClassVect

% %% Selecting Features By Genetic Algorithm
% ga;
% SelectedFeaturesIndices=BestSol.Out.S;
% load Features
% load ClassVect
% Features=Features(SelectedFeaturesIndices(1:end),:);
% 
% %% Training ANN
% % results=CreateAndTrainANN(Features,ClassVect);
% 
% %% Test ANN
% predict




%xlswrite('ClassVect.xlsx',ClassVect); % save ClassVect in an excle file
%xlswrite('Features.xlsx',Features);   % save Features in an excle file
