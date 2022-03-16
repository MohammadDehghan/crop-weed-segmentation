function [Accuracy,precision,Recall,F1score]=ComparingResults(g1,g2)

% g1=imread('C:\Users\mohammad\Desktop\project\annotations\022_annotation.png');
g1=imresize(g1,[619 697]);
% g2=imread('C:\Users\mohammad\Desktop\untitled.png');
g2=imresize(g2,[619 697]);
[r,c]=size(g1);
%% Labeling The Refrence Image
for i=1:r
    for j=1:c/3
        if g1(i,j,1)>0  % label red pixels 1
        g1(i,j,1)=1;
         
    elseif g1(i,j,2)>0  % label green pixels 2
        g1(i,j,2)=2;
        end
    end
end
%% Labeling The Output Image
for i=1:r
    for j=1:c/3
        if g2(i,j,1)>0   % label red pixels 1
           g2(i,j,1)=1;
             
      elseif g2(i,j,2)>0 % label green pixels 2
             g2(i,j,2)=2;
        end
    end
end


%% False Negetive And True Positives Pixels For Crop Class
FalseNAndTrueP=0;
for i=1:r
    for j=1:c/3
        if g1(i,j,2)==2  % the green ones in the ref. image 
            FalseNAndTrueP=FalseNAndTrueP+1;
            
        end
    end
end

%% True Negetive And False Positives Pixels For Crop Class
TrueNAndFalseP=0;
for i=1:r
    for j=1:c/3
        if g1(i,j,1)==1  % the red ones in the ref. image 
            TrueNAndFalseP=TrueNAndFalseP+1;
            
        end
    end
end
%% Calculating True Positive Pixels For Crop Class
TrueP=0;
for i=1:r
    for j=1:c/3
        if (g1(i,j,2)==2 && g2(i,j,2)==2)% the green ones in the Output image which are green in ref. image too
            TrueP=TrueP+1;
            
        end
    end
end

%% Calculating False Positive Pixels For Crop Class
FalseP=0;
for i=1:r
    for j=1:c/3
        if (g1(i,j,1)==1 && g2(i,j,2)==2 ) % the green ones in the Output image which should be red
            FalseP=FalseP+1;
            
        end
    end
end

%% Calculating precision For crop Class

TruePAndFalseP=TrueP+FalseP;
precision=TrueP/TruePAndFalseP;

if isnan(precision) % check if precision is nan or not
    precision=0;
end

%% Calculating Recall For crop Class
Recall=TrueP/FalseNAndTrueP;

%% Calculating F1score For crop Class
FalseN=FalseNAndTrueP-TrueP;
 F1score=(2*TrueP)/((2*TrueP)+FalseN+FalseP);
 
 %% Calculating Accuracy For crop Class
 TrueN=TrueNAndFalseP-FalseP;
 Accuracy=(TrueP+TrueN)/(TrueP+FalseN+FalseP+TrueN);
end