clc;close;clear;
%% Designate Folder for plot results
% Subfolder for plots
plotID = "Plots/";
mkdir Plots
%% Load in Data Inputs and Outputs 

IntableID = "../InTable_All_1700.txt";
OuttableID = "../OutTable_All_1700.txt";

In = readtable(IntableID);
Out_all = readtable(OuttableID);

Out = categorical(Out_all(:,19).Variables);

% Testing Percentage = 25%
p = 0.25;
cvpart = cvpartition(Out,'holdout',p);

In_train = In(training(cvpart),:);
Out_train = Out(training(cvpart),:);

In_test = In(test(cvpart),:);
Out_test = Out(test(cvpart),:);

% Normalize the Data
%[In_train_scale,Ctrain,Strain] = normalize(In_train);
%In_test_scale = normalize(In_test,"center",Ctrain,"scale",Strain);

InNames = In.Properties.VariableNames;
OutName = 'ErosionClass';
%% Set up the Model/Train the Model
Mdl = fitcensemble(In_train,Out_train,...
    'PredictorNames',InNames,...
    'ResponseName',OutName,...
    'OptimizeHyperparameters',{'NumLearningCycles','MaxNumSplits','LearnRate'}, ...
    'HyperparameterOptimizationOptions',struct('Repartition',true, ...
    'AcquisitionFunctionName','expected-improvement-plus')...
   );
%% Save the Model
%save("RF_class.mat",'Mdl');
Mdl = load("RF_class.mat");
%% Evaluate the Model and save the stats
Mdl = Mdl.Mdl;
Out_predict = Mdl.predict(In_test);

%% Plot and Save as .png files
figure
cm = confusionchart(Out_test,Out_predict);
cm.Title = 'RandF: Classification';
cm.FontSize = 20;
print('-dpng',plotID+"confusion_class.png")

%% Predictor Importance
imp = predictorImportance(Mdl);
[sorted_imp,isorted_imp] = sort(imp,'descend');
% isorted_imp has the index of the important predictors.  Save this
writematrix(isorted_imp,"Classification_imp.txt")

figure;
barh(imp(isorted_imp(1:20)));hold on;grid on;
barh(imp(isorted_imp(1:5)),'y');
barh(imp(isorted_imp(1:3)),'r');
title('Predictor Importance Classification');
xlabel('Estimates with Curvature Tests');ylabel('Predictors');
set(gca,'FontSize',20);
set(gca,'TickDir','out');
set(gca,'LineWidth',2);
ax = gca;ax.YDir='reverse';ax.XScale = 'log';
xlim([0.0 imp(isorted_imp(1))*1.2])
%ylim([.25 24.75])
% label the bars
for i=1:20%length(Mdl.PredictorNames)
    text(...
        1.05*imp(isorted_imp(i)),i,...
        strrep(Mdl.PredictorNames{isorted_imp(i)},'_',''),...
        'FontSize',14 ...
    )
end

print('-dpng',plotID+"Classification_input_importance_class.png")
%% Try again on the smaller dataset

Mdl = fitcensemble(In_train(:,isorted_imp(1:20)),Out_train,...
    'PredictorNames',InNames(isorted_imp(1:20)),...
    'ResponseName',OutName,...
    'OptimizeHyperparameters',{'NumLearningCycles','MaxNumSplits','LearnRate'}, ...
    'HyperparameterOptimizationOptions',struct('Repartition',true, ...
    'AcquisitionFunctionName','expected-improvement-plus')...
   );
%% Evaluate the Optimal-Input Model and save the stats

Out_predict = Mdl.predict(In_test(:,isorted_imp(1:20)));

%% Plot and Save as .png files
figure
cm = confusionchart(Out_test,Out_predict);
cm.Title = 'Erosion Class Detection (Reduced Input): RandomForest';
cm.FontSize = 20;
