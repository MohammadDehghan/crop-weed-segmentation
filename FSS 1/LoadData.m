function data=LoadData()

%     [x, t]=bodyfat_dataset();
    load C:\Users\mohammad\Desktop\project\MainFeatures.mat;
    data.x=MainFeatures;
 load C:\Users\mohammad\Desktop\project\ClassVect.mat;
      data.t= ClassVect;
    data.nx=size(data.x,1);
    data.nt=size(data.t,1);
    data.nSample=size(data.x,2);

end