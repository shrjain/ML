close all;
clear all;
Base = [0.8260;

0.8138;

0.8029;

0.7978];



m1 = [0.8221;

0.8132;

0.8021;

0.7943];



m2 = [

0.8255;

0.8122;

0.8078;

0.7978];



r = [3,5,7,10];



figure;

hold on;

plot(r,Base,'r.-','Linewidth',2);

plot(r,m1,'b.-','Linewidth',2);

plot(r,m2,'g.-','Linewidth',2);

xl = xlabel('Rank');

yl = ylabel('Prediction RMSE averaged over 20 trials');

LEG = legend('Baseline','Method-1','Method-2');

set(xl, 'FontSize', 14);
set(yl, 'FontSize', 14);
set(LEG, 'FontSize', 18);
ylim([0.793,0.827]);
set(gca,'fontsize',14)
print -f1 -djpeg -r600 MovieLens10MGenre.jpg