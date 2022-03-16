function data=LoadData()

%     [x, t]=bodyfat_dataset();
    load C:\Users\mohammad\Desktop\project\FSS2\Real\TN.mat;
    data.x=TN;
 load C:\Users\mohammad\Desktop\project\FSS2\Real\ClassVect.mat;
      data.t= ClassVect;
    data.nx=size(data.x,1);
    data.nt=size(data.t,1);
    data.nSample=size(data.x,2);

end