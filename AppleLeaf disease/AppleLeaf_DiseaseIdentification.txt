%%%%%IMAGE PREPROCESSING%%%%%%

clc;
close all;
clear all;
warning('off','all');

%%%%TO READ THE INPUT IMAGE FROM THE FILE %%%%

[filename,pathname]=uigetfile('*.jpg');
im=imread([pathname,filename]);

%%%%TO SHOW THE INPUT IMAGE%%%%
figure,imshow(im),title('INPUT IMAGE');

%%%TO RESIZE AND ADJUST THE INPUT IMAGE%%%

im=imresize(im,[256 256]);
im=imadjust(im,[]);
figure,imshow(im),title('RESIZED INPUT IMAGE');

%%PERFORM 3D-BOX FILTERING ON THE RESIZED IMAGE%%
vol = squeeze(im);
localMean = imboxfilt3(vol,[5 5 3]);
figure,imshow(localMean);
localMean=imadjust(localMean,[0.1 0.9],[0.1 0.9]);

figure,imshow(localMean),title('3D BOX-FILTERED IMAGE');

%PERFORM DECORRELATION ON THE BOX FILTERED IMAGE%%
B = decorrstretch(localMean);

figure,imshow(B),title('DECORRELATED IMAGE');

%%PERFORM 3D GAUSSIAN FILTERING%%
G = imgaussfilt3(B );
figure,imshow(G),title('3D GAUSSIAN FILTERED IMAGE');

%%3D MEDIAN FILTER%%
filter=medfilt3(G);
figure,imshow(filter);

%%%%%IMAGE SEGMENTATION%%%%%%%

%%HSI color transformation%%
HSV = rgb2hsv(filter);
figure,imshow(HSV);
H=HSV(:,:,1);
S=HSV(:,:,2);
V=HSV(:,:,3);
subplot(2,2,1), imshow(H),title('H-SPACE IMAGE');
subplot(2,2,2), imshow(S),title('S-SPACE IMAGE');
subplot(2,2,3), imshow(V),title('V-SPACE IMAGE');

%%%%MORPHOLOGICAL OPERATIONS%%%%%%
%%%%%SELECT THE S-CHANNEL IMAGE%%%%%

%%%CONVERT THE S-CHANNEL IMAGE TO BLACK AND WHITE IMAGE%%%%

bw=imbinarize(S,0.80);
figure,imshow(bw);
%%%%%TO REMOVE THE IMAGES IN THE SIZE OF 10 PIXELS%%%%%
%%%%%TO PERFORM MORPHOLOGOCAL OPENING (REMOVING SMALL AREA OF
PIXELS)%%%%%
bw1=bwareaopen(bw,10);
figure,imshow(bw1),title('SMALL OBJECTS REMOVED IMAGE');

%%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE
MATRIX)%%%%

g = graycomatrix(bw1);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(bw1);
Standard_Deviation = std2(bw1);
Entropy = entropy(bw1);
Skewness = skewness(bw1)
Variance = mean2(var(double(bw1)));
a = sum(double(bw1(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(bw1(:)));
Skewness = skewness(double(bw1(:)));

%%% Inverse Difference Movement%%%
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
for j = 1:n
temp = G(i,j)./(1+(i-j).^2);
in_diff = in_diff+temp;
end
end
IDM = double(in_diff);

feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation,
Entropy, Variance, Smoothness, Kurtosis, Skewness, IDM];

%% % Color Texture Feature Extraction(LBP) FOR TUMOR ALONE IMAGE%%%
% % % step 1: Local Binary Patterns
feat_LBP = extractLBPFeatures(bw1);
grayImage = bw1;
localBinaryPatternImage1 = zeros(size(grayImage));
[row col] = size(grayImage);
for r = 2 : row - 1
for c = 2 : col - 1
centerPixel = grayImage(r, c);
pixel7 = grayImage(r-1, c-1) > centerPixel;
pixel6 = grayImage(r-1, c) > centerPixel;
pixel5 = grayImage(r-1, c+1) > centerPixel;
pixel4 = grayImage(r, c+1) > centerPixel;
pixel3 = grayImage(r+1, c+1) > centerPixel;
pixel2 = grayImage(r+1, c) > centerPixel;
pixel1 = grayImage(r+1, c-1) > centerPixel;
pixel0 = grayImage(r, c-1) > centerPixel;
localBinaryPatternImage1(r, c) = uint8(...
pixel7 * 2^7 + pixel6 * 2^6 + ...
pixel5 * 2^5 + pixel4 * 2^4 + ...
pixel3 * 2^3 + pixel2 * 2^2 + ...
pixel1 * 2 + pixel0);
end
end

figure,imshow(localBinaryPatternImage1),title('LBP IMAGE');
G1=im2double(feat_LBP);

%%%FEATURE EXTRACTION BY GLCM (GRAY LEVEL CO-OCCURENCE MATRIX)
FOR LBP IMAGE%%%%
g1 = graycomatrix(G1);
stats1 = graycoprops(g1,'Contrast Correlation Energy Homogeneity');
Contrast1 = stats.Contrast;
Correlation1 = stats.Correlation;
Energy1 = stats.Energy;
Homogeneity1 = stats.Homogeneity;
Mean1 = mean2(G1);
Standard_Deviation1 = std2(G1);
Entropy1 = entropy(G1);
Skewness1 = skewness(G1)
Variance1 = mean2(var(double(G1)));
a1 = sum(double(G1(:)));
Smoothness1 = 1-(1/(1+a));
Kurtosis1 = kurtosis(double(G1(:)));
Skewness1 = skewness(double(G1(:)));

%%%% Inverse Difference Movement %%%%
m1 = size(G1,1);
n1 = size(G1,2);
in_diff = 0;
for i = 1:m1
for j = 1:n1
temp = G1(i,j)./(1+(i-j).^2);
in_diff1 = in_diff+temp;
end
end
IDM1 = double(in_diff1);
feat_disease2 = [Contrast1,Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1,
Entropy1, Variance1, Smoothness1, Kurtosis1, Skewness1, IDM1];

%% ADD THE TWO FEATURES %%
feat_disease=([feat_disease1+feat_disease2]);
load featureapple.mat

%%%% KNN CLASSIFICATION%%%%
label=ones(1,108);
label(1:31)=1;
label(32:51)=2;
label(52:74)=3;
label(75:108)=4;

feat=feature_finalapple; label=label;
% % % (Method 1) GA
close all;
N=10; T=100; CR=0.8; MR=0.01;
[sFeat,Sf,Nf,curve]=jGA(feat,label,N,T,CR,MR);
%
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('GA'); grid on;

% save 'sFeat.mat' sFeat

feat_disease11=[Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,
Variance, Smoothness,IDM];
feat_disease22=[Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1, Entropy1,
Variance1, Smoothness1,IDM1];
feat_disease333=([feat_disease11+feat_disease22]);

load sFeat.mat

label=ones(1,108);
label(1:31)=1;
label(32:51)=2;
label(52:74)=3;
label(75:108)=4;

model=fitcknn(sFeat,label);
result=predict(model,feat_disease333);

if result == 1
msgbox(' APPLE RUST');
disp(' APPLE RUST ');
elseif result == 2
msgbox(' APPLE SCAB ');
disp('APPLE SCAB');
elseif result == 3
msgbox('BLOCK ROT');
disp('BLOCK ROT');
elseif result == 4
msgbox('HEALTHY');
disp('HEALTHY');
end