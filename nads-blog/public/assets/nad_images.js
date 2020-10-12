const fig_nad_images = function () {
    let network = 'ln';

    const graphs_width = 300;
    const graphs_height = 300;
    const graphs_margin = { top: 10, right: 70, bottom: 30, left: 40 };
    const graphs_offset_x = 0;

    const lines_svg = d3
        .select('#line-plots')
        .append('svg')
        .attr('width', graphs_width + graphs_margin.right)
        .attr('height', graphs_height);

    const graph_eigs = lines_svg
        .append('g')
        .attr(
            'transform',
            'translate(' +
                (graphs_offset_x + graphs_margin.left) +
                ',' +
                graphs_margin.top +
                ')'
        );

    const accs_offset_y = 150;
    const graph_accs = lines_svg
        .append('g')
        .attr(
            'transform',
            'translate(' +
                (graphs_offset_x + graphs_margin.left) +
                ',' +
                (graphs_margin.top + accs_offset_y) +
                ')'
        );

    const nad_size = 150;
    const nads_width = 350;
    const nads_height = 300;

    const nad_vis = d3
        .select('#nad-zoom')
        .append('svg')
        .attr('width', nads_width)
        .attr('height', nads_height);

    nad_img = show_nad(nad_vis, nad_size, 20, 90, 10, 'ln', 'nad');
    fourier_nad_img = show_nad(
        nad_vis,
        nad_size,
        nad_size + 50,
        90,
        10,
        'ln',
        'fourier_nad'
    );

    nad_vis
        .append('rect')
        .attr('width', 5)
        .attr('height', 200)
        .attr('x', 0)
        .attr('y', 40)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', d3.color('rgba(0,0,0,0.5)'))
        .style('stroke', 'none');

    const single_graph_height = 100;
    focus_acc = plot_accuracies(
        graph_accs,
        graphs_width,
        single_graph_height,
        network
    );
    plot_eigenvalues(
        graph_eigs,
        graphs_width,
        single_graph_height,
        network,
        focus_acc,
        nad_img,
        fourier_nad_img
    );

    decorate_select_network(
        'select_net_nads',
        'network_name_nads',
        function (network) {
            focus_acc = plot_accuracies(
                graph_accs,
                graphs_width,
                single_graph_height,
                network
            );
            plot_eigenvalues(
                graph_eigs,
                graphs_width,
                single_graph_height,
                network,
                focus_acc,
                nad_img,
                fourier_nad_img
            );
        },
        network
    );
};

const show_nad = function (selection, size, x, y, index, network, prefix) {
    selection
        .append('svg:image')
        .attr('width', size)
        .attr('height', size)
        .attr('x', x)
        .attr('y', y)
        .attr(
            'xlink:href',
            'images/nads/' + network + '/' + prefix + '_' + index + '.png'
        )
        .attr('id', prefix + '_img');

    selection
        .append('rect')
        .attr('width', size)
        .attr('height', size)
        .attr('x', x)
        .attr('y', y)
        .attr('rx', 20)
        .attr('ry', 20)
        .attr('fill', 'none')
        .style('stroke', 'white')
        .style('stroke-width', '16px');

    return selection.select('#' + prefix + '_img');
};

const plot_eigenvalues = function (
    selection,
    width,
    height,
    network,
    focus_acc,
    nad_img,
    fourier_nad_img
) {
    selection.selectAll('g').remove();
    selection.selectAll('path').remove();
    d3.csv('data/nads.csv', function (data) {
        const x = d3.scaleLinear().domain([0, 1024]).range([10, width]);
        let eig_net = 'eig_' + network;
        let max_eig = data[0][eig_net];
        let min_eig = data[1022][eig_net];

        selection
            .append('g')
            .attr('transform', 'translate(0,' + height + ')')
            .attr('class', 'axis')
            .call(d3.axisBottom(x).tickValues([0, 1023]).tickSize(0));            

        // Add Y axis
        const y = d3
            .scaleLog()
            .domain([min_eig / max_eig, 1])
            .range([height, 0]);        

        let orders_magnitude = Math.ceil(Math.log10(max_eig / min_eig));
        let yticks = [];
        for (i = 0; i < orders_magnitude; i++) {
            yticks[i] = Math.pow(10, -i);
        }

        const y_axis = selection
            .append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).tickValues(yticks).tickSize(0))            
            .selectAll('.tick text')                       
            .text(null)
            .text(10)
            .append('tspan')                
            .attr('dy', '-.7em')
            .attr('font-size', '7px')
            .text(function (d) {
                return Math.round(Math.log(d) / Math.LN10);
            });

        // Add the line
        selection
            .append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', d3.color('#0E726A'))
            .attr('stroke-width', 1.5)
            .attr(
                'd',
                d3
                    .line()
                    .x(function (d) {
                        return x(d.index);
                    })
                    .y(function (d) {
                        return y(d[eig_net] / max_eig);
                    })
            );
        const focus_line = selection
            .append('g')
            .append('line')
            .attr('x1',0).attr('y1',0)
            .attr('x2', 100).attr('y2',100)
            .style('stroke', d3.color('#333'))
            .style('opacity', 0);
 
        const focus_eig = selection
            .append('g')
            .append('circle')
            .style('fill', d3.color('#0E726A') )
            .attr('stroke-width', 0)
            .attr('r', 2.5)
            .style('opacity', 0);
       // Create a rect on top of the svg area: this rectangle recovers mouse position
        const mouse_active_area = selection
            .append('rect')
            .style('fill', 'none')
            .style('pointer-events', 'all')
            .attr('width', width)
            .attr('height', 2*height+50)
            .on('mouseover', mouseover)
            .on('mousemove', mousemove)
            .on('mouseout', mouseout);

        // What happens when the mouse move -> show the annotations at the right positions.
        function mouseover() {
            focus_eig.style('opacity', 1);
            focus_acc.style('opacity', 1);
            focus_line.style('opacity', 1);
        }

        const y_acc = d3.scaleLog().domain([50, 100]).range([height, 0]);

        function mousemove() {
            // recover coordinate we need
            console.log(d3.mouse(this)[0])
            var x0 = x.invert(d3.mouse(this)[0]);
            var i = bisect(data, x0, 0);
            selectedData = data[i];

            focus_line
                .attr('x1', x(selectedData.index)).attr('y1', 0)
                .attr('x2', x(selectedData.index)).attr('y2', mouse_active_area.attr('height'));
            focus_eig
                .attr('cx', x(selectedData.index))
                .attr('cy', y(selectedData[eig_net] / max_eig));
    
            const clean_acc = function (d) {
                    return y_acc(d['acc_' + network]);
            };

            focus_acc
                .attr('cx', x(selectedData.index))
                .attr('cy', clean_acc(selectedData));

            nad_img.attr(
                'xlink:href',
                'images/nads/' + network + '/nad_' + selectedData.index + '.png'
            );
            fourier_nad_img.attr(
                'xlink:href',
                'images/nads/' +
                    network +
                    '/fourier_nad_' +
                    selectedData.index +
                    '.png'
            );
            d3.select('#show-nad-idx').text(selectedData.index);
        }
        function mouseout() {
            focus_acc.style('opacity', 0);
            focus_eig.style('opacity', 0);
            focus_line.style('opacity', 0);
        }
    });
};

const plot_accuracies = function (selection, width, height, network) {
    selection.selectAll('g').remove();
    selection.selectAll('path').remove();

    d3.csv('data/nads.csv', function (data) {
        const x = d3.scaleLinear().domain([0, 1024]).range([10, width]);
        const acc_net = 'acc_' + network;

        selection
            .append('g')
            .attr('transform', 'translate(0,' + height + ')')
            .attr('class', 'axis')
            .call(d3.axisBottom(x).tickValues([0, 1023]).tickSize(0));

        // Add Y axis
        const y = d3.scaleLog().domain([50, 100]).range([height, 0]);

        const y_axis = selection.append('g')
            .attr('class', 'axis')
            .call(
            d3
                .axisLeft(y)
                .tickValues([50, 100])
                .tickFormat((d) => d + '%')
                .tickSize(0)
        )

        // Add the line
        selection
            .append('path')
            .datum(
                data.filter(function (d) {
                    return d[acc_net].length > 0;
                })
            )
            .attr('fill', 'none')
            .attr('stroke', d3.color('#B17423'))
            .attr('stroke-width', 1.5)
            .attr(
                'd',
                d3
                    .line()
                    .x(function (d) {
                        return x(d.index);
                    })
                    .y(function (d) {
                        return y(d[acc_net]);
                    })
            );
    });

    const focus_acc = selection
        .append('g')
        .append('circle')
        .style('fill', d3.color('#B17423'))
        .attr('stroke-width', 0)
        .attr('r', 2.5)
        .style('opacity', 0);
    return focus_acc;
};

const update_nad_curves = function (acc_svg, eig_svg, network) {};

// This allows to find the closest X index of the mouse:
const bisect = d3.bisector(function (d) {
    return d.index;
}).left;

fig_nad_images();
