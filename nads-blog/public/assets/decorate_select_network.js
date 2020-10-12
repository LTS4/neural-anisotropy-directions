const decorate_select_network = function (select_id, network_name_id, update_fn, default_network) {
    document.getElementById(
        select_id
    ).value = default_network; /*select default option*/

    //adjust size of select for default option
    d3.select('#' + select_id).style(
        'width',
        d3.select('#' + network_name_id).node().offsetWidth + 'px'
    );

    d3.select('#' + select_id).on('change', function () {
        network = this.options[this.selectedIndex].value;

        //adjust size of select using a dummy element
        network_name = this.options[this.selectedIndex].text;
        d3.select('#' + network_name_id).text(network_name);
        d3.select(this).style(
            'width',
            d3.select('#' + network_name_id).node().offsetWidth + 'px'
        );

        update_fn(network);
    });
};

const get_select_value = function(select_id) {
    let select_object = d3.select('#' + select_id).node()
    console.log(select_object)
    return select_object.options[select_object.selectedIndex].value;
}
