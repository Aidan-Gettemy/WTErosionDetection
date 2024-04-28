clc;close;clear
mkdir('Plots')
saveID = 'Plots/';
IntableID = "../InTable_All_1700.txt";
OuttableID = "../OutTable_All_1700.txt";
In = readtable(IntableID);
Out = readtable(OuttableID);
In.TurbineAge = double(Out(:,"TurbineAgeClass").Variables);

figure
pltname = "";
scatter(In,"B1N3Cd","TipDyb3","filled","ColorVariable",616)
colormap turbo
c = colorbar;
c.Label.String = 'Turbine Age (0-24 years)';
c.Location = 'eastoutside';
xscale log
fontsize(16,"points")

%yscale log

%saveID = saveID+pltname+".png;
%print('-dpng',saveID)
