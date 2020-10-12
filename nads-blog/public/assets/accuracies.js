/* eslint-disable no-undef */
const fig_accuracies = function () {
    const dft_svg = d3
        .select('#dft')
        .append('svg')
        .attr('position', 'absolute')
        .style('left', '0px')
        .attr('height', 400)
        .attr('width', 400);

    const vec_svg = d3
        .select('#zoom')
        .append('svg')
        .attr('position', 'absolute')
        .style('left', '0px')
        .attr('height', 200)
        .attr('width', 250);

    vec_svg
        .append('rect')
        .attr('width', 5)
        .attr('height', 170)
        .attr('x', 10)
        .attr('y', 10)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', d3.color('rgba(0,0,0,0.5)'))
        .style('stroke', 'none');

    const lines_svg = d3
        .select('#acc-div')
        .append('svg')
        .style('width', 1000 + 'px')
        .style('height', 1000 + 'px')
        .style('position', 'absolute')
        .style('left', '0px')
        .lower();

    const l1 = lines_svg
        .append('line')
        .style('stroke', d3.color('#777777'))
        .style('stroke-width', '1.5px')
        .attr('x1', 0)
        .attr('x2', 400)
        .attr('y1', 0)
        .attr('y2', 200)
        .style('visibility', 'hidden');

    const l2 = lines_svg
        .append('line')
        .style('stroke', d3.color('#777777'))
        .style('stroke-width', '1.5px')
        .attr('x1', 0)
        .attr('x2', 400)
        .attr('y1', 0)
        .attr('y2', 200)
        .style('visibility', 'hidden');

    let network = 'ln';

    const dist_svg = d3.select('#dist_svg');

    plot_dft(dft_svg, dist_svg, 280, network, l1, l2);
    plot_dir(vec_svg, 65, 45, 120, 10, 5);
    plot_distribution(dist_svg, 10, 0, 200, 150);
};

const plot_dft = function (dft_svg, dist_svg, size, network, l1, l2) {
    const dft_offset_x = 80;
    const dft_offset_y = 10;

    const dft_g = dft_svg
        .append('g')
        .attr(
            'transform',
            'translate(' + dft_offset_x + ',' + dft_offset_y + ')'
        )
        .attr('id', 'dft_g');
    d3.csv('data/dft_accuracies.csv', function (data) {
        // Labels of row and columns -> unique identifier of the column called 'group' and 'variable'
        const x_coords = d3
            .map(data, function (d) {
                return d.x;
            })
            .keys();
        const y_coords = d3
            .map(data, function (d) {
                return d.y;
            })
            .keys();

        // Build X scales and axis:
        const x = d3.scaleBand().range([0, size]).domain(x_coords).padding(0);

        // Build Y scales and axis:
        const y = d3.scaleBand().range([size, 0]).domain(y_coords).padding(0);

        var cubehelix = d3.interpolateCubehelix("rgb(29,23,61)", "rgb(204,216,176)");

        // Build color scale
        const myColor = d3
            .scaleSequential()
            .interpolator(cubehelix)
            .domain([50, 100]);

        const opacity_dft = 0.6;

        // Three function that change the tooltip when user hover / move / leave a cell
        const mouseover_dft = function (d, network, object) {
            dft_g.selectAll('rect').style('opacity', opacity_dft);
            update_test_accuracy(Math.round(d['acc_' + network]));

            d3.select(object)
                .attr('width', x.bandwidth() * 1.4)
                .attr('height', y.bandwidth() * 1.4)
                .attr(
                    'transform',
                    'translate(' +
                        -x.bandwidth() * 0.2 +
                        ',' +
                        -y.bandwidth() * 0.2 +
                        ')'
                )
                .style('stroke', 'black')
                .style('opacity', 1.0)
                .raise();

            update_distribution(dist_svg);

            update_dir(d.x, d.y);
            d3.select('#zoom').transition().duration(300).style('opacity', 1.0);

            l1.attr('x1', dft_offset_x + size)
                .attr('y1', y(d.y) + dft_offset_y + y.bandwidth() * 0.5)
                .style('visibility', 'visible');

            l2.attr('x2', l1.attr('x1')).attr('y2', l1.attr('y1'));

            l2.attr(
                'x1',
                x(d.x) +
                    x.bandwidth() * 1.4 +
                    dft_offset_x -
                    x.bandwidth() * 0.2
            )
                .attr('y1', l1.attr('y1'))
                .style('visibility', 'visible');
        };

        const mouseleave = function (_d) {
            d3.select(this)
                .attr(
                    'transform',
                    'translate(' +
                        x.bandwidth() * 0.0 +
                        ',' +
                        y.bandwidth() * 0.0 +
                        ')'
                )
                .style('stroke', 'none')
                .style('opacity', opacity_dft)
                .attr('width', x.bandwidth() - 0.2)
                .attr('height', y.bandwidth() - 0.2)
                .lower();
        };

        dft_g
            .selectAll()
            .data(data, function (d) {
                return d.x + ':' + d.y;
            })
            .enter()
            .append('rect')
            .attr('x', function (d) {
                return x(d.x);
            })
            .attr('y', function (d) {
                return y(d.y);
            })
            .attr('width', x.bandwidth() - 0.2)
            .attr('height', y.bandwidth() - 0.2)
            .style('fill', function (d) {
                return myColor(d['acc_' + network]); /*default network: LeNet*/
            })
            .style('stroke-width', 2)
            .style('opacity', 1.0)
            .on('mouseover', function (d) {
                mouseover_dft(d, network, this);
            })
            .on('mouseleave', mouseleave);

        decorate_select_network(
            'select_net_dft',
            'network_name_dft',
            function (network) {
                dft_svg
                    .selectAll('rect')
                    .on('mouseover', function (d) {
                        mouseover_dft(d, network, this);
                    })
                    .style('fill', (d) => myColor(d['acc_' + network]));
            },
            network
        );
    });

    const mouseleave_dft = function (_d) {
        d3.select(this)
            .selectAll('rect')
            .transition()
            .duration(300)
            .style('opacity', 1.0);

        d3.select('#zoom').transition().duration(300).style('opacity', 0.3);
        l1.transition().duration(300).style('visibility', 'hidden');
        l2.transition().duration(300).style('visibility', 'hidden');
    };

    dft_g.on('mouseleave', mouseleave_dft);

    var cubehelix = d3.interpolateCubehelix("rgb(29,23,61)", "rgb(204,216,176)");

    const colorBar = renderColorBar(
        dft_svg,
        cubehelix,
        55,
        10,
        10,
        size
    );

    colorBar
        .append('text')
        .attr('x', -33)
        .attr('y', 10)
        .attr('text-anchor', 'right')
        .style('font-size', '10px')
        .text('100%');

    colorBar
        .append('text')
        .attr('x', -30)
        .attr('y', size)
        .attr('text-anchor', 'right')
        .style('font-size', '10px')
        .text('50%');
};
const plot_distribution = function (dist_svg, x, y, dist_width, dist_height) {
    const cx_neg = dist_width / 5;
    const cx_pos = (4 * dist_width) / 5;
    const numSamples = 40;
    const variance = 30;
    const jitter = 3;
    const radius = 2.5;
    const rotation = 0;

    const dist_g = dist_svg
        .append('g')
        .attr('transform', 'translate(' + x + ',' + y + ')');

    for (i = 0; i < numSamples; i++) {
        dist_g
            .append('circle')
            .style('fill', d3.color('#B17423'))
            .attr('cx', cx_neg + d3.randomNormal(0, jitter)())
            .attr('cy', d3.randomNormal(dist_height / 2, variance))
            .attr('r', radius)
            .attr('opacity', 1)
            .attr(
                'transform',
                'rotate(' +
                    rotation +
                    ',' +
                    dist_width / 2 +
                    ',' +
                    dist_height / 2 +
                    ')'
            )
            .attr('id', 'pos_point');
    }

    for (i = 0; i < numSamples; i++) {
        dist_g
            .append('circle')
            .style('fill', d3.color('#0E726A'))
            .attr('cx', cx_pos + d3.randomNormal(0, jitter)())
            .attr('cy', d3.randomNormal(dist_height / 2, variance))
            .attr('r', radius)
            .attr('opacity', 1)
            .attr(
                'transform',
                'rotate(' +
                    rotation +
                    ',' +
                    dist_width / 2 +
                    ',' +
                    dist_height / 2 +
                    ')'
            )
            .attr('id', 'neg_point');
    }
};

const update_distribution = function (dist_svg) {
    const variance = 30;
    const dist_height = 150;
    dist_svg
        .selectAll('circle')
        .transition()
        .duration(300)
        .attr('cy', d3.randomNormal(dist_height / 2, variance));
};

const plot_dir = function (svg, x, y, size, k_x, k_y) {
    const vec_svg = svg
        .append('g')
        .attr('transform', 'translate(' + x + ',' + y + ')');

    const vec_image = vec_svg
        .append('svg:image')
        .attr('width', size)
        .attr('height', size)
        .attr('x', 0)
        .attr('y', 0)
        .attr('xlink:href', 'images/fourier_' + k_x + '_' + k_y + '.png')
        .attr('id', 'vec_image');

    vec_svg
        .append('rect')
        .attr('width', size)
        .attr('height', size)
        .attr('x', 0)
        .attr('y', 0)
        .attr('rx', 12)
        .attr('ry', 12)
        .attr('fill', 'none')
        .style('stroke', 'white')
        .style('stroke-width', '15px');
};
const update_dir = function (k_x, k_y) {
    d3.select('#vec_image').attr(
        'xlink:href',
        'images/fourier_' + k_x + '_' + k_y + '.png'
    );
};

const update_test_accuracy = function (acc) {
    d3.select('#test_acc').text(acc);
};

fig_accuracies();
