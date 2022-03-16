cd 'C:\Users\mohammad\Desktop\project\images' % go to images folder to get input images
ImgList4=dir; % take a list from the current diroctory
cd .. % come back to before dir

cd 'C:\Users\mohammad\Desktop\project\masks' % go to masks folder to get mask images
Img_list2=dir; % take a list from the current diroctory
cd .. % come back to before dir

cd 'C:\Users\mohammad\Desktop\project\annotations' % go to annotations folder to get ground truth images
Img_list3=dir;% take a list from the current diroctory
cd .. % come back to before dir
    
n=80; %size of each block
% load  TNp.mat
counter=1;
FinalImg=zeros(800,800,3);
% Features2=zeros(1,209);

%% Reading The Images 
for i4=21:60
fulname4=ImgList4(i4+2).name;  % because the first and second elemnts of Img_list are cd. and cd.. folders we begine with the 23th element of Img_list
cd 'C:\Users\mohammad\Desktop\project\images' % go to images folder to get input images
MainImg=imread(fulname4); % read the image
% MainImg=rgb2gray(MainImg);  % convort rgb imge to gray one
MainImg=imresize(MainImg,[810 810]); % adjust the image size
[r, c]=size(MainImg);
BlockedImg=zeros(n); % initialization
cd ..

fullname2=Img_list2(i4+2).name;  % because the first and second elemnts of Img_list are cd. and cd.. folders we begine with the 23th element of Img_list
cd 'C:\Users\mohammad\Desktop\project\masks' % go to masks folder to get mask images
MaskImg=imread(fullname2); % read the image
cd ..
MaskImg=imresize(MaskImg,[810 810]);


fullname3=Img_list3((2*i4)+1).name;  % because the first and second elemnts of Img_list are cd. and cd.. folders we begine with the 23th element of Img_list
cd 'C:\Users\mohammad\Desktop\project\annotations'
LabeledImg=imread(fullname3); % read the image
LabeledImg=imresize(LabeledImg,[810 810]); % adjust the image size
cd ..

LabeledImg2=LabeledImg; % copy the LabeledImg

FinalImg=zeros(810,810,3); % initialization

%% Removing The Background Of Image
MaskImg=imresize(MaskImg,[810 810]);
for i1=1:r
    for j1=1:c/3
        if MaskImg(i1,j1,:)==1
            
            MainImg(i1,j1,:)=0;
        end
    end
end
% figure();imshow(MainImg)
%% Blocking The Image Into n*n block
for i5=1:80:810
    for j5=1:80:810
        if i5+(n-1)<=810 && j5+(n-1)<=810
           
         BlockedImg2=MainImg(i5:i5+(n-1),j5:j5+(n-1),:);
         BlockedImg=rgb2gray(BlockedImg2);
       BlockedImg=im2double(BlockedImg);

%% Extracing Features
if nnz(BlockedImg)>=200 % to prevent to enter whole black blocks
%           figure();imshow(BlockedImg2)
%           figure();imshow(BlockedImg)
          
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

Im_gray=BlockedImg; % copy BlockedImg
Im_med=medfilt2(Im_gray); % median filtering 
Im_bw=im2bw(Im_med,graythresh(Im_med)); % thresholding
% if sum(sum(Im_bw))/numel(Im_bw)>0.5
%     Im_bw=~Im_bw;
% end

prop=regionprops(Im_bw); % measure properties of image regions
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
 t = statxture (BlockedImg(Im_bw)); % t(4)= skewness of histogram

% % Feature 1
% min_image = min(BlockedImg(Im_bw));% min of the block
% % Feature 2
% mean_image = mean(BlockedImg(Im_bw));% mean of the block
% % Feature 3
% max_image = max(BlockedImg(Im_bw));% max of the block
for i=1:80
    for j=1:80
        if BlockedImgGlcm(i,j)==0
            BlockedImgGlcm(i,j)=NaN;
        end
    end
end
 warning('off', 'Images:graycomatrix:scaledImageContainsNan')

glcm = graycomatrix(BlockedImgGlcm,'Offset',[2 0]);
stats = graycoprops(glcm);
Contrast=stats.Contrast;
Correlation=stats.Correlation;
Energy=stats.Energy;
Homogeneity=stats.Homogeneity;


%     % zernike moment n=order and m=repition
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
%     arr(19)=real(mom);
%     arr(20)=imag(mom);
%     arr(21)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,2,0);      % Call Zernikemoment fuction n=2, m=0
%     arr(22)=real(mom);
%     arr(23)=imag(mom);
%     arr(24)=amplitude;
%    clear mom amplitude angle
%    [mom, amplitude, angle] = Zernikmoment(Im_bw,2,2);      % Call Zernikemoment fuction n=2, m=2
%     arr(25)=real(mom);
%     arr(26)=imag(mom);
%     arr(27)=amplitude;
%    clear mom amplitude angle
LBP = extractLBPFeatures(BlockedImg,'Upright',false);
% HOG = extractHOGFeatures(BlockedImg);
[M]=feature_vec(BlockedImg);

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
max_imagess= max(reshape(BlockedImg,[r*c,1]));% max of the block

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

v2=var(reshape(BlockedImgNonZero,[NumNonzero,1]));
s2=std(reshape(BlockedImgNonZero,[NumNonzero,1]));
k=kurtosis(reshape(BlockedImgNonZero,[NumNonzero,1]));

% TNp2=TNp(counter,:);

Features1=[max_image,min_image,mean_image,max_imagess,min_imagess,mean_imagess,max_imagev,min_imagev,mean_imagev,k,LBP,Contrast,Correlation,Energy,Homogeneity]; % ghathering max,min,mean features together
Features2=[Ar,premiter,convex_hull,major,minor,major_minor,compactness,solidity,convexity]; % ghathering Regionprops features together

%s=[159,78,25,328,387,150,320,73,181,449,168,149,358,259,137,249,63,441,309,459,438,10,362,446,444,37,398,295,395,479,319,430,23,308,176,421,90,71,427,108,227,224,222,351,167,270,97,187,238,173,59,341,451,4,360,189,483,464,491,276,254,161,32,293,477,481,3,145,359,114,412,288,439,89,499,490,84,347,332,185];

% selectedTn=TNp2(:,s(1:end));
MainFeatures=[Features1,Features2,t(4),]'; % ghathering all features together
% MainFeaturesSelectedI=[2,3,4,6,11,22,26,27,28,29,31,32,33]; % main
% features selected by ga

% MainFeaturesSelected=MainFeatures(MainFeaturesSelectedI(1:end),1);
% selectedTn=selectedTn';
% Features=[Features;selectedTn];
% indicesFeaturesSelected=[23 15 17 19 24 25 10 6 7 12 9 29 20 5 8 18 4 11 21 30];
% FeaturesSelected2= Features(indicesFeaturesSelected(1:end),1);

% indicesFeaturesSelected=[6 27 40 37 34 8 17 3 19 15 14 22 12 10 9 29 25 31 20 28 18 24 11 4 30 41 1 35 21 13];
% FeaturesSelected2= Features(indicesFeaturesSelected(1:end),1);


%% Detecting The Class Of Block

Detect=sim(results.net,MainFeatures);

a=zeros(80,80,3);% initialization
    if (max(Detect) == Detect(1)) % If the block belongs to class 1 ?
          
         for i=1:80
             for j=1:80
                 if BlockedImg(i,j)~=0
                     a(i,j,1)=255;
                     a(i,j,2)=0;
                     a(i,j,3)=0;
                 
                 
                 end
             end
         end


  elseif (max(Detect) == Detect(2)) % If the block belongs to class 2 ?

          for i22=1:80
             for j22=1:80
                 if BlockedImg(i22,j22)~=0
                     a(i22,j22,1)=0;
                     a(i22,j22,2)=255;
                     a(i22,j22,3)=0;
                     
                
                 end
             end
         end

    end
     
%% Construct The Output image     
     
FinalImg(i5:i5+(80-1),j5:j5+(80-1),:)=a;
counter=counter+1;
 end      
      
        end
   end
end

%% Assessment The Output Image

[y1 ,y2 ,y3 ,y4]=ComparingResults(LabeledImg2,FinalImg); % calculate Avrage Acuuracy,Percision,Recall,F1score of each output images respectively by calling ComparingResults
ComparedResults(:,i4-20)=[y1 ,y2 ,y3 ,y4]';
  
%% Writing The Output Images

% cd 'C:\Users\mohammad\Desktop\project\Output images6' % go to blocked image folder
% imwrite(FinalImg,[num2str(i4),'.png']); % write the image
% cd ..
  end

%% Assessment Whole Output Images

AvrageAcuuracy=mean(ComparedResults(1,:)) % cal Avrage Acuuracy of whole output images
Percision=mean(ComparedResults(2,:)) % cal Percision of whole output images
Recall=mean(ComparedResults(3,:)) % cal Recall of whole output images
F1score=mean(ComparedResults(4,:)) % cal F1score of whole output images

%      pause(.5)
%  figure();imshow(FinalImg) 
 

