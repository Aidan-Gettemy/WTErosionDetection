clc;close;clear
mkdir('Plots')
saveID = 'Plots/';
IntableID = "../InTable_All_1700.txt";
OuttableID = "../OutTable_All_1700.txt";
In = readtable(IntableID);
Out = readtable(OuttableID);
In.TurbineAge = double(Out(:,"TurbineAgeClass").Variables);

rankingsID = "../FeatureSelection/Classification_imp.txt";

M = readmatrix(rankingsID);close

Names = In.Properties.VariableNames;

for i = 1:20
    a = i; b = i+1;
    figure
    pltname = "";
    scatter(In,Names{M(a)},Names{M(b)},"filled","ColorVariable",616)
    colormap turbo
    c = colorbar;
    c.Label.String = 'Turbine Age (0-24 years)';
    c.Location = 'eastoutside';
    if min(In(:,Names{M(a)}).Variables)>0
        xscale log
    else
        xscale linear
    end
    fontsize(16,"points")
    ttl = Names{M(a)}+" vs "+Names{M(b)} + " (Ranks ["+num2str(a)+...
        ", "+num2str(b)+"])";
    title(ttl)
    if min(In(:,Names{M(b)}).Variables)>0
        yscale log
    else
        yscale linear
    end
end
%% Plot and Save
pts = [9,13;12,13;19,20;17,18;16,17;51,367;15,16;13,14;118,119];
for i = 1:numel(pts(:,1))
    figure
    a = pts(i,1);b = pts(i,2);
    scatter(In,Names{M(a)},Names{M(b)},"filled","ColorVariable",616)
    colormap turbo
    c = colorbar;
    c.Label.String = 'Turbine Age (0-24 years)';
    c.Location = 'eastoutside';
    if min(In(:,Names{M(a)}).Variables)>0
        xscale log
    else
        xscale linear
    end
    fontsize(16,"points")
    ttl = Names{M(a)}+" vs "+Names{M(b)} + " (Ranks ["+num2str(a)+...
        ", "+num2str(b)+"])";
    title(ttl)
    if min(In(:,Names{M(b)}).Variables)>0
        yscale log
    else
        yscale linear
    end
    pltname = "classimp"+num2str(a)+"vs"+num2str(b);
    pltsaveID = saveID+pltname+".png";
    print('-dpng',pltsaveID)
end
%% Plot a scatter in 3D
a = 13;b = 15; c1 = 16;
f = figure;
f.Position = [100 100 800 600];
scatter3(In,Names{M(a)},Names{M(b)},Names{M(c1)},'filled',"ColorVariable",616)
colormap turbo
c = colorbar;
c.Label.String = 'Turbine Age (0-24 years)';
c.Location = 'eastoutside';
fontsize(16,"points")
ttl = Names{M(a)}+" vs "+Names{M(b)} + " vs "+Names{M(c1)};
grid on
title(ttl)
if min(In(:,Names{M(b)}).Variables)>0
    yscale log
else
    yscale linear
end
if min(In(:,Names{M(a)}).Variables)>0
    xscale log
else
    xscale linear
end
if min(In(:,Names{M(c1)}).Variables)>0
    zscale log
else
    zscale linear
end
pltname = "";
pltname = "classimp"+num2str(a)+"vs"+num2str(b)+"vs"+num2str(c1);
pltsaveID = saveID+pltname+".png";
print('-dpng',pltsaveID)
