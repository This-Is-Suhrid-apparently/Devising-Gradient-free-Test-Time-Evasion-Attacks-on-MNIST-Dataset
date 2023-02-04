function [z,acc_list] = zoo_stoch(net, testX, testY, adv_labels_map)

learn_rate = 0.01;
testX = double(testX);

z = 0.01*rand(28,28,1); %double(zeros(28,28,1));10*rand(28,28,1);
testX_corr = zeros(size(testX));
for i = 1:numel(testY)
    testX_corr(:,:,:,i) = ((testX(:,:,:,i))+z);
end


%cross-entropy from f and labels
F = predict(net,testX);
sum_F_class = sum(F(1,:));
L = -log( F(1, (testY(1)+1) )/sum_F_class ) ;
%acc threshold for convergence criterion
acc_thr = 0.8;

%this will give the predicted labels 
predLabelsTest = net.classify(testX); %_corr); %testX_corr);
%calc acc
accuracy = sum(predLabelsTest == categorical(transpose(testY))) / numel(testY);

coord_map = randperm((28*28));

acc_list = [];
accuracy = 0.0;

while(accuracy<acc_thr)
    for i = 1:(28*28)
        x = mod(coord_map(i),28)+1;
        disp(i);
        y = min(floor(coord_map(i)/28)+1,28);
        L=0; L_corr = 0;
        for j = 1:numel(testY) 
            X_corr = testX;
            try
                X_corr(x, y,:,j) = testX( x, y,:,j) + z(x,y,:);
            catch
                disp(x), disp(y), disp(j)
            end
            F_corr = predict(net,X_corr(:,:,:,j));
            sum_F_class = sum(F_corr(:,:));
            L_corr = L_corr - log( F_corr(:, (adv_labels_map(j) +1) )/sum_F_class ) ;
            F = predict(net,testX(:,:,:,j));
            sum_F_class = sum(F(:,:));
            L = L - log( F(:,(testY(j)+1) ) /sum_F_class );
        end
        gi = (L_corr-L)/(z(x,y,:));
        z(x,y,:) = ( (z(x,y,:)) + learn_rate*gi ) ;

    testX_corr = testX;
    for i = 1:numel(testY)
        testX_corr(:,:,:,i) = (double(testX(:,:,:,i))+z(:,:,:));
    end
    predLabelsTest = net.classify(testX_corr);  
    end
    %this will give the predicted labels 
    testX_corr = testX;
    for i = 1:numel(testY)
        testX_corr(:,:,:,i) = (double(testX(:,:,:,i))+z(:,:,:));
    end
    predLabelsTest = net.classify(testX_corr);
    %calc acc
    accuracy = sum(predLabelsTest == categorical(transpose(adv_labels_map))) / numel(adv_labels_map);
    accuracy %disp(accuracy)
    acc_list = [acc_list, accuracy];
end
