/* eslint-disable no-undef */
const fig_poison = function () {
    d3.selectAll(("input[name='train_set']")).on("change", on_check)
};

const on_check = function () {
        if (this.value == 'poison'){                
            d3.select('#dog_class').attr('href', 'images/poisoning-dog-poison.png')
            d3.select('#cat_class').attr('href', 'images/poisoning-cat-poison.png')
            d3.select('#poison-explanation').style('visibility', 'visible')    
        }
        else {
            d3.select('#dog_class').attr('href', 'images/poisoning-dog.png')
            d3.select('#cat_class').attr('href', 'images/poisoning-cat.png')

            d3.select('#poison-explanation').style('visibility', 'hidden')
        }
}

const on_check_old = function () {
    const poison_number_1 = d3.select('#poison_number_1')
    const poison_number_2 = d3.select('#poison_number_2')
    const poison_number_3 = d3.select('#poison_number_3')
    const poison_number_4 = d3.select('#poison_number_4')

    const poison_highlight_1 = d3.select('#poison_highlight_1')
    const poison_highlight_2 = d3.select('#poison_highlight_2')
    const poison_highlight_3 = d3.select('#poison_highlight_3')
    const poison_highlight_4 = d3.select('#poison_highlight_4')

    if (this.value == 'poison'){
        d3.selectAll('.coefficient').transition().duration(500).style('opacity', 0.3)
        d3.selectAll('.nad.first').transition().duration(500).style('opacity', 0.3)
        d3.selectAll('.nad.second').transition().duration(500).style('opacity', 0.3)
        d3.selectAll('.clip').transition().duration(500).style('opacity', 1)
        d3.selectAll('.number').transition().duration(500).style('opacity', 0.05)
        d3.select('#poison-explanation').style('visibility', 'visible')

        d3.select('#dog_1').attr('href', 'images/156_dog_poison.png')
        d3.select('#dog_2').attr('href', 'images/dog_poison.png')
        d3.select('#cat_1').attr('href', 'images/266_cat_poison.png')
        d3.select('#cat_2').attr('href', 'images/479_cat_poison.png')

        poison_number_1.transition().duration(500).text('0.5\\;')
        poison_number_2.transition().duration(500).text('0.5\\;')
        poison_number_3.transition().duration(500).text('-0.5')
        poison_number_4.transition().duration(500).text('-0.5')

        poison_highlight_1.attr('class', 'number poison_dog').transition().duration(500).style('opacity', 0.2)
        poison_highlight_2.attr('class', 'number poison_dog').transition().duration(500).style('opacity', 0.2)
        poison_highlight_3.attr('class', 'number poison_cat').transition().duration(500).style('opacity', 0.2)
        poison_highlight_4.attr('class', 'number poison_cat').transition().duration(500).style('opacity', 0.2)

    } else {
        d3.selectAll('.coefficient').transition().duration(500).style('opacity', 1)
        d3.selectAll('.number').transition().duration(500).style('opacity', 0.15)
        d3.selectAll('.nad.first').transition().duration(500).style('opacity', 1)
        d3.selectAll('.nad.second').transition().duration(500).style('opacity', 1)
        d3.select('#poison-explanation').style('visibility', 'hidden')

        d3.select('#dog_1').attr('href', 'images/156_dog.png')
        d3.select('#dog_2').attr('href', 'images/dog.png')
        d3.select('#cat_1').attr('href', 'images/266_cat.png')
        d3.select('#cat_2').attr('href', 'images/479_cat.png')

        poison_number_1.transition().duration(500).text('4.15')
        poison_number_2.transition().duration(500).text('-1.6')
        poison_number_3.transition().duration(500).text('-12')
        poison_number_4.transition().duration(500).text('2.83')
        
        poison_highlight_1.attr('class', 'number')
        poison_highlight_2.attr('class', 'number')
        poison_highlight_3.attr('class', 'number')
        poison_highlight_4.attr('class', 'number')
    }
}

fig_poison();
