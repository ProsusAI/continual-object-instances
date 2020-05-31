

mkdir images

names = dir('cars/*.mat');


for i = 1:numel(names)
    
    data = load(fullfile('cars', names(i).name));
    
    allims = data.im;
    
    mkdir(fullfile('images', [names(i).name(1:end-4)]));
    
    for j = 1:24
        
        for k = 1:4
            
            im = allims(:,:,:,j,k);
            
            savename = fullfile('images', [names(i).name(1:end-4)], [names(i).name(1:end-4) '_' int2str(j) '_' int2str(k) '.jpg']);
            imwrite(im, savename);
            
        end
        
    end
    
end

