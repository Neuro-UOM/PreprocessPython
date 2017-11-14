files = dir( './*.mat' );
names = { files.name };
%names(1) = [];names(1) = [];names(1) = [];names(1) = [];
for i = 1:size(names,2)
    file_name = names(i);
    file_name = char(file_name);
    %disp(class(char(file_name)));
    data = load(file_name);
    data = data.EEGdata;
    transposed = data';
    csvwrite(strcat(file_name,'.csv'),transposed)
end