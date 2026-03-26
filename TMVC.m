function Y = TMVC(X,A,Z,gnd,q,lam1,lam2,lam3,d,p,eta)

NITER = 20;
nV = length(X);                % view number
[N ,dv]= size(X{1});

k = length(unique(gnd));

C1 = cell(nV,1);            % di * d
C2 = cell(nV,1);            % d  * q
QI = eye(q);

for v = 1:nV
    [~,dv]= size(X{v});
    IV{v} = eye(dv);
    C1{v} = zeros(N,d);
    C2{v} = zeros(N,q);
%     AZ{v} = Z{v}*A{v};
%     AA{v} = A{v}*A{v}';
    E{v} = zeros(N,d);   
end

%% Initialize
mu  = 1e-5;  max_mu  = 10e10; 
rho = 1e-5;  max_rho = 10e10;
% eta = 2;

sX = [N q nV];
Z_tensor = cat(3, Z{:,:});    % tensorized
for iter = 1: NITER
    %% updating tensor J
    for jv =1:nV
        Z = Z_tensor(:,:,jv);
        BB{jv}=(Z + C2{jv}/rho);
    end
    B_tensor = cat(3,BB{:,:});
    Bt = B_tensor(:);
    [j, ~] = wshrinkObj_weight_lp(Bt,lam2/rho,sX, 0,3,p);
    J_tensor = reshape(j,sX);       

    %% updating tensor W    
    for ww = 1:nV
        Z = Z_tensor(:,:,ww);
        L1 = (E{ww}-C1{ww}/mu)' * (X{ww}- Z*A{ww});
        XZX = (X{ww}- Z*A{ww})' * (X{ww}- Z*A{ww});
        QQ = computingqq(X{ww},A{ww},Z);
        L2 = 2*QQ + lam1*IV{ww} + mu*XZX;
        W{ww} = mu*L1*inv(L2);
    end
    
    %% update Z   
    for vv = 1:nV
        J = J_tensor(:,:,vv);
        WY = EuDist2(X{vv}*W{vv}',A{vv}*W{vv}',0);
        L3 = mu*A{vv}*W{vv}'*W{vv}*A{vv}' + rho*QI;
        Tem1 = A{vv}*W{vv}'*C1{vv}';
        Tem2 = mu*A{vv}*W{vv}'*(W{vv}*X{vv}'-E{vv}');
        Tem3 =  rho*J-C2{vv}-WY;
        L4 = Tem1 + Tem2 + Tem3';
        FF = inv(L3)*L4;
        Z_tensor(:,:,vv) = FF';
    end

    %% updating E    
    for ei = 1:nV
       Z = Z_tensor(:,:,ei);
        EE1 = W{ei}*X{ei}' -  W{ei}*A{ei}'*Z'; 
        temp = EE1'+C1{ei}/mu';
        E{ei} = solve_l1l2(temp,lam3/mu);
        
    end

     

    %% updating C1，C2
    for cc = 1:nV
        tt1 = W{cc}*X{cc}' -  W{cc}*A{cc}'*Z' - E{cc}';
        C1{cc} = C1{cc} + mu*tt1';
        Z = Z_tensor(:,:,cc);
        J = J_tensor(:,:,cc);
        tt2 = Z - J;
        C2{cc} = C2{cc} + rho*tt2;
    end

    mu  = min(eta*mu, max_mu);
    %rho = mu;
    rho = min(eta*rho, max_rho);

end

Sbar = [];
for uu = 1:nV
    Z = abs(Z_tensor(:,:,uu));
    Sbar = cat(1,Sbar,1/sqrt(nV)*Z');
end
[G,Sig,H] = mySVD(Sbar',k); 
Y = G./repmat(sqrt(sum(G.^2,2)),1,k);

end

function QQ = computingqq(X,A,Z)
dd = sum(Z,2);
% This type of calculation only requires O (dn) for storage, without the need to store n × n matrices.
YDY = (X' .* dd') * X;     
ee = sum(Z,1);
ZEZ = (A' .* ee) * A;  
QQ = double(YDY -2*A'*Z'*X + ZEZ);
end
