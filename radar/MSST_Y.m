function [Ts,tfr1] = MSST_Y(x,hlength,num)
% Computes the first order MSST (Ts)  of the signal x.
% Expression (21)-based Algorithm.
% INPUT
%    x      :  Signal needed to be column vector.
%    hlength:  The hlength of window function.
%    num    :  iteration number.
% OUTPUT
%    Ts     :  The SST
%    tfr1     :  The STFT

[xrow,xcol] = size(x);

hlength=hlength+1-rem(hlength,2);  % 如果是偶数就加1变奇数，如果是奇数就还是奇数
ht = linspace(-0.5,0.5,hlength);ht=ht'; % ht是hlength个-0.5到0.5的数组

% Gaussian window
h = exp(-pi/0.32^2*ht.^2);  % h是高斯窗

[hrow,~]=size(h); Lh=(hrow-1)/2; % 这tm用size?? hrow窗长,Lh是半窗长

N=xrow; %原始数据长度
t=1:xrow; %这是干什么,数数的?

[~,tcol] = size(t); % 这句话写的十分的fuck, tcol=时间长度点数

tfr= zeros (round(N/2),tcol) ; % trf是一个行为半原始数据长度,列为原始数据长度,woc这是提前设计好的频率范围吗?
omega = zeros (round(N/2),tcol-1);% omega是一个行半原始数据长度,列为原始数据长度减1

for icol=1:tcol  % 这是时间长度遍历
    ti= t(icol); %j8编程思维
    tau=-min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti]);
    indices= rem(N+tau,N)+1; 
    rSig = x(ti+tau,1);
    tfr(indices,icol)=rSig.*conj(h(Lh+1+tau));
end  % 总之这里就是在分割加窗

tfr=fft(tfr);

tfr=tfr(1:round(N/2),:);  % 取有用部分前一半
 
tfr1=tfr;  % 这句话没用
for i=1:round(N/2)
omega(i,:)=diff(unwrap(angle(tfr(i,:))))*(N)/2/pi;  % 这里有差分
end
omega(:,end+1)=omega(:,end);
omega=round(omega);  % 取整数部分

for ite=1:num  % 这里是其迭代部分,那么上面全都是准备工作?
[Ts]=SST(tfr,omega);
tfr=Ts;
end

Ts=Ts/(xrow/2);  % 对输出又进行了幅度处理
end

function [Ts_f]=SST(tfr_f,omega_f);
[tfrm,tfrn]=size(tfr_f);
Ts_f= zeros (tfrm,tfrn) ; 
%mx=max(max(tfr_f));
for b=1:tfrn%time  时间遍历
    % Reassignment step
    for eta=1:tfrm%frequency  频率遍历
        %if abs(tfr_f(eta,b))>0.001*mx%you can set much lower value than this.
            k = omega_f(eta,b);
            if k>=1 && k<=tfrm
                Ts_f(k,b) = Ts_f(k,b) + tfr_f(eta,b);
            end
        %end
    end
end
end